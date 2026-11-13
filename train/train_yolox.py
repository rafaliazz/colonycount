"""Train the YOLOX-Tiny / -M baseline on the RF-DETR tiles (task 2.3). Apache-2.0, shippable.

Usage: uv sync --extra train --group dev --group yolox, then
       uv run --no-sync python train/train_yolox.py --size tiny|m [--epochs N] [--batch B]
       uv run --no-sync python train/train_yolox.py --size m --resume <run dir>/weights/last.pth

The YOLOX model code is vendored in train/third_party/yolox (pinned commit, see its NOTICE); the
PyPI and git packages don't install on Python 3.14. YOLOX reads COCO, so the make_tiles.py tiles
in <data-dir>/public/tiles_<T>/{train,valid} are used as they are, at their native size T (edge
tiles zero-padded bottom/right, as colonycount does at inference), single class. Starts from the
COCO weights (downloaded once to <data-dir>/weights/, SHA-256 checked); the 80-class classifier is
replaced by a 1-class one.

Same budget as RF-DETR and YOLOv8: COLONYCOUNT_TRAIN_EPOCHS, seed COLONYCOUNT_SEED, early stopping
after 5 epochs without a better val mAP@0.5:0.95, effective batch nbs = COLONYCOUNT_TRAIN_BATCH_SIZE
x COLONYCOUNT_TRAIN_GRAD_ACCUM (16) reached by accumulating max(round(nbs / batch), 1) steps of
--batch tiles (default COLONYCOUNT_TRAIN_BATCH_SIZE). Recipe stays YOLOX's own: SGD (momentum 0.9,
nesterov, wd 5e-4 on conv weights), lr 0.01 / 64 per image of the effective batch, quadratic warmup
then cosine to 5%, EMA of the weights (validated and saved as "ema"), AMP on CUDA.

Augmentation is YOLOX's default too (yolox/exp/yolox_base.py): TrainTransform's augment_hsv
(hsv_prob 1.0, gains h 5 / s 30 / v 30) then a horizontal flip (flip_prob 0.5), vendored from
yolox/data/data_augment.py and applied, as YOLOX does, to the cv2-loaded BGR uint8 tile (the ONNX
export feeds the model BGR x 255 too). Mosaic/mixup/affine are YOLOX's first-phase extras, off for
the last no_aug_epochs (15) epochs, which is the whole of a <= 15-epoch run; that no-aug phase
also adds YOLOX's L1 box loss (trainer.before_epoch: head.use_l1). Longer runs are refused (see
mosaic_epochs). TrainTransform's max_labels=120 cap is not copied: it drops ground truth on dense
tiles and is a data-loading limit, not augmentation.

Writes <data-dir>/runs/yolox_<size>/<timestamp>/: results.csv (train losses, val
metrics/mAP50(B) and metrics/mAP50-95(B) per epoch, as Ultralytics names them), weights/last.pth
(every epoch) + best.pth, summary.json (config, "comparable", wall time, peak VRAM). --resume
continues in the same run dir with the checkpoint's own settings; only --workers can change.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import time
from pathlib import Path

import numpy as np
from dotenv import load_dotenv

from colonycount._env import env_int, env_path

SIZES = {"tiny": (0.33, 0.375), "m": (0.67, 0.75)}  # depth, width (YOLOX exps/default)
WEIGHTS_URL = "https://github.com/Megvii-BaseDetection/YOLOX/releases/download/0.1.1rc0/"
WEIGHTS_SHA256 = {
    "tiny": "9de513de589ac98bb92d3bca53b5af7b9acfa9b0bacb831f7999d0f7afaee8f0",
    "m": "60076992b32da82951c90cfa7bd6ab70eba9eda243e08b940a396f60ac2d19b6",
}
LR_PER_IMG = 0.01 / 64  # YOLOX basic_lr_per_img
WARMUP_EPOCHS = 1  # YOLOX: 5 of 300; one epoch here, of 12
MIN_LR_RATIO = 0.05
EMA_DECAY = 0.9998
FLIP_PROB, HSV_PROB = 0.5, 1.0  # YOLOX Exp flip_prob, hsv_prob
HSV_GAINS = (5, 30, 30)  # augment_hsv's default hgain, sgain, vgain (TrainTransform uses those)
NO_AUG_EPOCHS = 15  # YOLOX Exp no_aug_epochs: last epochs without mosaic/mixup, with L1 loss
PATIENCE = 5
EVAL_CONF, EVAL_NMS, MAX_DET = 0.001, 0.65, 300  # YOLOX eval settings; max_det as YOLOv8's
COLUMNS = [
    "epoch",
    "train/iou_loss",
    "train/conf_loss",
    "train/cls_loss",
    "metrics/mAP50(B)",
    "metrics/mAP50-95(B)",
    "lr/pg0",
    "time",
]


def load_split(split_dir: Path) -> tuple[list[tuple[Path, int, np.ndarray]], dict]:
    """[(jpg, COCO image id, labels [N, 5] as class 0, cx, cy, w, h in pixels)], and the COCO
    dict (the val ground truth)."""
    coco = json.loads((split_dir / "_annotations.coco.json").read_text())
    boxes: dict[int, list] = {i["id"]: [] for i in coco["images"]}
    for a in coco["annotations"]:
        x, y, w, h = a["bbox"]
        boxes[a["image_id"]].append([0, x + w / 2, y + h / 2, w, h])
    items = [
        (split_dir / i["file_name"], i["id"], np.array(boxes[i["id"]], np.float32).reshape(-1, 5))
        for i in coco["images"]
    ]
    return items, coco


def accumulation(nbs: int, batch: int) -> int:
    """Steps per optimizer step to reach the effective batch nbs (Ultralytics' rule)."""
    return max(round(nbs / batch), 1)


def lr_at(step: int, total: int, warmup: int, lr: float) -> float:
    """YOLOX's warmcos per optimizer step: quadratic warmup from 0, then cosine to MIN_LR_RATIO."""
    if step < warmup:
        return lr * ((step + 1) / warmup) ** 2  # YOLOX starts at 0; never a no-op step
    t = (step - warmup) / max(total - warmup, 1)
    return lr * (MIN_LR_RATIO + (1 - MIN_LR_RATIO) * 0.5 * (1 + math.cos(math.pi * t)))


def mosaic_epochs(epochs: int, no_aug_epochs: int = NO_AUG_EPOCHS) -> int:
    """Epochs YOLOX would train with mosaic/mixup before its no-aug phase; 0 is all we support."""
    n = max(epochs - no_aug_epochs, 0)
    # ponytail: only YOLOX's no-aug phase is implemented, so runs > 15 epochs are refused; for
    # them vendor MosaicDetection (mosaic, mixup, random_affine) for the first n epochs.
    if n:
        raise ValueError(
            f"{epochs} epochs: YOLOX's default recipe would use mosaic/mixup for the first {n} "
            f"(no_aug_epochs {no_aug_epochs}); that phase isn't implemented, use <= {no_aug_epochs}"
        )
    return n


def should_stop(maps: list[float], patience: int = PATIENCE) -> bool:
    """True after `patience` epochs without a better val mAP@0.5:0.95 than the best before."""
    best = maps.index(max(maps))
    return len(maps) - 1 - best >= patience


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", choices=list(SIZES), required=True, help="YOLOX-Tiny or YOLOX-M")
    ap.add_argument("--data-dir", type=Path, help="default: COLONYCOUNT_DATA_DIR")
    ap.add_argument("--epochs", type=int, help="default: COLONYCOUNT_TRAIN_EPOCHS")
    ap.add_argument(
        "--batch", type=int, help="tiles per step; default: COLONYCOUNT_TRAIN_BATCH_SIZE"
    )
    ap.add_argument("--fraction", type=float, default=1.0, help="share of train tiles to use")
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--device", default="cuda", help="cuda or cpu (falls back to cpu)")
    ap.add_argument("--resume", type=Path, help="weights/last.pth of a stopped run to continue")
    args = ap.parse_args(argv)
    try:
        args.data_dir = args.data_dir or env_path("COLONYCOUNT_DATA_DIR", "~/data/colonycount")
        if args.epochs is None:
            args.epochs = env_int("COLONYCOUNT_TRAIN_EPOCHS", 12, lo=1)
        mosaic_epochs(args.epochs)
        if args.batch is None:
            args.batch = env_int("COLONYCOUNT_TRAIN_BATCH_SIZE", 2, lo=1)
    except ValueError as e:
        ap.error(str(e))
    if args.resume is not None and not args.resume.is_file():
        ap.error(f"--resume: no checkpoint at {args.resume}")
    return args


def build_config(args: argparse.Namespace) -> dict:
    """The run's settings (JSON-able; saved in every checkpoint and summary.json)."""
    tile = env_int("COLONYCOUNT_TRAIN_TILE_SIZE", 640, lo=1)
    nbs = env_int("COLONYCOUNT_TRAIN_BATCH_SIZE", 2, lo=1) * env_int(
        "COLONYCOUNT_TRAIN_GRAD_ACCUM", 8, lo=1
    )
    return dict(
        size=args.size,
        tiles=str(args.data_dir / "public" / f"tiles_{tile}"),
        out=str(args.data_dir / "runs" / f"yolox_{args.size}" / time.strftime("%Y%m%d-%H%M%S")),
        epochs=args.epochs,
        imgsz=tile,
        seed=env_int("COLONYCOUNT_SEED", 0),
        patience=PATIENCE,
        nbs=nbs,  # effective batch, as RF-DETR's batch x grad accum
        batch=args.batch,  # per step
        accumulate=accumulation(nbs, args.batch),
        lr=LR_PER_IMG * nbs,
        fraction=args.fraction,
    )


def comparable(config: dict, amp: bool) -> dict:
    """The settings that must match RF-DETR's and YOLOv8's runs."""
    keys = ("epochs", "imgsz", "seed", "patience", "nbs", "batch", "accumulate")
    c = {k: config[k] for k in keys} | {"optimizer": "SGD", "amp": amp, "ema": EMA_DECAY}
    c["recipe"] = dict(
        flip_prob=FLIP_PROB,
        hsv_prob=HSV_PROB,
        hsv_gains=list(HSV_GAINS),
        mosaic_epochs=mosaic_epochs(config["epochs"]),
        no_aug_epochs=NO_AUG_EPOCHS,
        l1_loss=True,
    )
    print("comparable config:", json.dumps(c))
    if config["batch"] > config["nbs"]:
        print(f"WARNING: per-step batch {config['batch']} > nbs {config['nbs']}: no accumulation")
    return c


class Tiles:
    """Map-style dataset: BGR 0..255 float32 [3, T, T] (YOLOX's input), labels [N, 5].
    augment: YOLOX's TrainTransform augmentation (HSV, then horizontal flip)."""

    def __init__(self, items, tile: int, augment: bool):
        self.items, self.tile, self.augment = items, tile, augment

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import cv2

        path, image_id, labels = self.items[i]
        img, labels = cv2.imread(str(path)), labels.copy()  # BGR uint8, as YOLOX reads it
        h, w = img.shape[:2]
        if self.augment:  # DataLoader seeds `random` and numpy per worker
            from third_party.yolox.data_augment import _mirror, augment_hsv

            if random.random() < HSV_PROB:
                augment_hsv(img, *HSV_GAINS)
            c, half = labels[:, 1:3], labels[:, 3:5] / 2
            img, xyxy = _mirror(img, np.hstack([c - half, c + half]), FLIP_PROB)
            labels[:, 1:3] = (xyxy[:, :2] + xyxy[:, 2:]) / 2
        out = np.zeros((3, self.tile, self.tile), np.float32)
        out[:, :h, :w] = img.transpose(2, 0, 1)
        return out, labels, image_id


def collate(batch):
    """Stack images; zero-pad labels to the most boxes in the batch (YOLOX skips all-zero rows)."""
    import torch

    imgs, labels, ids = zip(*batch, strict=True)
    n = max(1, *(len(lb) for lb in labels))
    targets = np.zeros((len(batch), n, 5), np.float32)
    for k, lb in enumerate(labels):
        targets[k, : len(lb)] = lb
    return torch.from_numpy(np.stack(imgs)), torch.from_numpy(targets), list(ids)


def build_model(size: str, num_classes: int = 1):
    """YOLOX-<size> as YOLOX's exps build it (BN eps 1e-3, momentum 0.03; prior-prob biases)."""
    import torch
    from third_party.yolox import YOLOPAFPN, YOLOX, YOLOXHead

    depth, width = SIZES[size]
    in_channels = [256, 512, 1024]
    model = YOLOX(
        YOLOPAFPN(depth, width, in_channels=in_channels),
        YOLOXHead(num_classes, width, in_channels=in_channels),
    )
    for m in model.modules():
        if isinstance(m, torch.nn.BatchNorm2d):
            m.eps, m.momentum = 1e-3, 0.03
    model.head.initialize_biases(1e-2)
    return model


def load_pretrained(model, size: str, weights_dir: Path) -> list[str]:
    """Load the COCO weights where shapes match (all but the class head); return skipped keys."""
    import torch

    path = weights_dir / f"yolox_{size}.pth"
    if not path.is_file():
        weights_dir.mkdir(parents=True, exist_ok=True)
        torch.hub.download_url_to_file(f"{WEIGHTS_URL}yolox_{size}.pth", str(path))
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != WEIGHTS_SHA256[size]:
        raise SystemExit(f"{path}: SHA-256 {digest} is not the released file's")
    state = torch.load(path, map_location="cpu", weights_only=True)["model"]
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and own[k].shape == v.shape}
    model.load_state_dict(keep, strict=False)
    return sorted(set(own) - set(keep))


def evaluate(model, loader, coco_gt: dict, device) -> tuple[float, float]:
    """(mAP@0.5, mAP@0.5:0.95) on the val tiles with pycocotools, up to MAX_DET per tile."""
    import contextlib
    import io

    import torch
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
    from third_party.yolox import postprocess

    model.eval()
    dets = []
    with torch.no_grad():
        for imgs, _, ids in loader:
            out = postprocess(model(imgs.to(device)).float(), 1, EVAL_CONF, EVAL_NMS)
            for image_id, d in zip(ids, out, strict=True):
                if d is None:
                    continue
                d = d[:MAX_DET].cpu().numpy()  # NMS output is sorted by score
                for x1, y1, x2, y2, obj, cls, _ in d:
                    dets.append(
                        dict(
                            image_id=image_id,
                            category_id=1,
                            bbox=[float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                            score=float(obj * cls),
                        )
                    )
    if not dets:
        return 0.0, 0.0
    with contextlib.redirect_stdout(io.StringIO()):
        gt = COCO()
        gt.dataset = coco_gt
        gt.createIndex()
        e = COCOeval(gt, gt.loadRes(dets), "bbox")
        e.params.maxDets = [MAX_DET]  # pycocotools' 100 would cap the densest tiles
        e.evaluate()
        e.accumulate()
    p = e.eval["precision"][:, :, 0, 0, -1]  # [IoU 0.5:0.95, recall], class 0, area all
    return float(p[0][p[0] > -1].mean()), float(p[p > -1].mean())  # what summarize() averages


def train(config: dict, args: argparse.Namespace, ckpt: dict | None) -> dict:
    import torch
    from third_party.yolox import ModelEMA

    seed = config["seed"]
    random.seed(seed)
    np.random.seed(seed)  # augment_hsv (workers=0; DataLoader workers reseed numpy themselves)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda" if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    amp = device.type == "cuda"
    out = Path(config["out"])
    (out / "weights").mkdir(parents=True, exist_ok=True)

    tiles = Path(config["tiles"])
    train_items, _ = load_split(tiles / "train")
    val_items, coco_val = load_split(tiles / "valid")
    if config["fraction"] < 1:
        rng = random.Random(seed)
        train_items = rng.sample(train_items, max(1, round(len(train_items) * config["fraction"])))
    loader_args = dict(
        batch_size=config["batch"],
        num_workers=args.workers,
        collate_fn=collate,
        persistent_workers=args.workers > 0,
        pin_memory=amp,
    )
    gen = torch.Generator().manual_seed(seed)
    train_loader = torch.utils.data.DataLoader(
        Tiles(train_items, config["imgsz"], augment=True),
        shuffle=True,
        drop_last=False,
        generator=gen,
        **loader_args,
    )
    val_loader = torch.utils.data.DataLoader(
        Tiles(val_items, config["imgsz"], augment=False), **loader_args
    )

    model = build_model(config["size"])
    if ckpt is None:
        skipped = load_pretrained(model, config["size"], args.data_dir / "weights")
        print(f"COCO weights loaded; new: {skipped}")
    else:
        model.load_state_dict(ckpt["model"])
    model.head.use_l1 = True  # YOLOX adds L1 in its no-aug phase, which is the whole run here
    model.to(device)
    ema = ModelEMA(model, EMA_DECAY)

    pg0, pg1, pg2 = [], [], []  # BN weights, conv weights (decayed), biases: YOLOX's groups
    for m in model.modules():
        if hasattr(m, "bias") and isinstance(m.bias, torch.nn.Parameter):
            pg2.append(m.bias)
        if isinstance(m, torch.nn.BatchNorm2d):
            pg0.append(m.weight)
        elif hasattr(m, "weight") and isinstance(m.weight, torch.nn.Parameter):
            pg1.append(m.weight)
    opt = torch.optim.SGD(pg0, lr=config["lr"], momentum=0.9, nesterov=True)
    opt.add_param_group({"params": pg1, "weight_decay": 5e-4})
    opt.add_param_group({"params": pg2})
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    acc = config["accumulate"]
    per_epoch = math.ceil(len(train_loader) / acc)
    total, warmup = config["epochs"] * per_epoch, WARMUP_EPOCHS * per_epoch
    start, maps, step = 0, [], 0
    if ckpt is not None:
        ema.ema.load_state_dict(ckpt["ema"])
        ema.updates = ckpt["ema_updates"]
        opt.load_state_dict(ckpt["optimizer"])
        scaler.load_state_dict(ckpt["scaler"])
        start, maps, step = ckpt["epoch"] + 1, ckpt["maps"], ckpt["step"]
        gen.manual_seed(seed + start)  # not epoch 1's tile order and augmentation again
        random.seed(seed + start)
        np.random.seed(seed + start)
    results = out / "results.csv"
    if not results.exists():
        results.write_text(",".join(COLUMNS) + "\n")

    t0 = time.time()
    for epoch in range(start, config["epochs"]):
        if maps and should_stop(maps):
            break
        model.train()
        sums, n = np.zeros(3), 0
        opt.zero_grad()
        for i, (imgs, targets, _) in enumerate(train_loader):
            imgs, targets = imgs.to(device, non_blocking=True), targets.to(device)
            with torch.autocast(device.type, enabled=amp):
                losses = model(imgs, targets)
            scaler.scale(losses["total_loss"] / acc).backward()
            sums += [float(losses[k].detach()) for k in ("iou_loss", "conf_loss", "cls_loss")]
            n += 1
            if (i + 1) % acc == 0 or i + 1 == len(train_loader):  # last partial group too
                lr = lr_at(step, total, warmup, config["lr"])
                for g in opt.param_groups:
                    g["lr"] = lr
                scaler.step(opt)
                scaler.update()
                opt.zero_grad()
                ema.update(model)
                step += 1
            if i % 200 == 0:
                print(
                    f"epoch {epoch + 1}/{config['epochs']} step {i}/{len(train_loader)} "
                    f"loss {float(losses['total_loss']):.3f} {time.time() - t0:.0f}s",
                    flush=True,
                )
        map50, map5095 = evaluate(ema.ema, val_loader, coco_val, device)
        maps.append(map5095)
        row = [epoch + 1, *(sums / max(n, 1)), map50, map5095, opt.param_groups[0]["lr"]]
        with results.open("a", newline="") as f:
            csv.writer(f).writerow([f"{v:.5g}" for v in row] + [round(time.time() - t0)])
        print(f"epoch {epoch + 1}: val mAP50 {map50:.4f} mAP50-95 {map5095:.4f}", flush=True)
        state = dict(
            size=config["size"],
            config=config,
            epoch=epoch,
            step=step,
            maps=maps,
            model=model.state_dict(),
            ema=ema.ema.state_dict(),
            ema_updates=ema.updates,
            optimizer=opt.state_dict(),
            scaler=scaler.state_dict(),
        )
        torch.save(state, out / "weights" / "last.pth")
        if map5095 == max(maps):
            torch.save(state, out / "weights" / "best.pth")
    return dict(
        device=device.type,
        amp=amp,
        best_map50_95=max(maps) if maps else None,
        epochs_run=len(maps),
        peak_vram_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2) if amp else None,
        wall_s=round(time.time() - t0),
    )


def main() -> None:
    load_dotenv(".env")  # from the working directory; real environment variables win
    args = parse_args()

    import torch  # train extra only; CI tests the config without it

    ckpt = None
    if args.resume is not None:
        ckpt = torch.load(args.resume, map_location="cpu", weights_only=True)
        config = ckpt["config"]  # the run's own settings, epochs included
    else:
        config = build_config(args)
    resolved = comparable(config, amp=args.device == "cuda" and torch.cuda.is_available())
    stats = train(config, args, ckpt)
    summary = {"config": config, "comparable": resolved, **stats}
    name = "summary_resumed.json" if ckpt is not None else "summary.json"
    (Path(config["out"]) / name).write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
