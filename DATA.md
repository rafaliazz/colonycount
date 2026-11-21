# Data

Datasets used to train and benchmark colonycount, their licenses, and the attribution they require.
The datasets are not in this repository; download them from the sources below.

## Datasets

| | Smartphone set | Clinical set | AGAR |
|---|---|---|---|
| Title | Annotated dataset for deep-learning-based bacterial colony detection | Clinical Dataset of 19 Bacterial Colony Images | Annotated Germs for Automated Recognition (AGAR) |
| Host | Figshare | Zenodo | NeuroSYS (sign-up required) |
| DOI / URL | [10.6084/m9.figshare.22022540](https://doi.org/10.6084/m9.figshare.22022540) (v3) | [10.5281/zenodo.18256864](https://doi.org/10.5281/zenodo.18256864) | <https://agar.neurosys.com> |
| Images | 369 | 950 original (1,900 with augmented copies) | 18,000 |
| Colonies (boxes) | 56,865 | 59,221 original (118,442 with augmented copies) | 336,442 |
| Species | 24 (veterinary) | 19 (clinical, 151 isolates) | 5 microorganisms |
| Files used | all 377 files, 641,992,356 bytes (642 MB) | `images.zip` (5,567,573,405 B) + `label.zip` (2,649,065,613 B) + `Bacterial_Image_Records.xlsx` (41,174 B) = 8,216,680,192 bytes (8.2 GB) | full set (size not stated on the public page) |
| License | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/) | [CC BY-NC 2.0 Generic](https://creativecommons.org/licenses/by-nc/2.0/) (non-commercial) |
| License verified at | Figshare API `api.figshare.com/v2/articles/22022540` (`license`), 2026-11-12 | Zenodo API `zenodo.org/api/records/18256864` (`metadata.license` = `cc-by-4.0`), 2026-11-12 | agar.neurosys.com landing page ("freely available for academic research under Creative Common Attribution-NonCommercial 2.0 Generic license"), 2026-11-12 |

Counts and file sizes come from the same API records and pages, read on 2026-11-12.

## Attribution

### Smartphone set (CC BY 4.0)

Dataset: Solymosi, N. & Nagy, S. Á. (2023). *Annotated dataset for deep-learning-based bacterial colony detection*. figshare. Dataset. <https://doi.org/10.6084/m9.figshare.22022540.v3>. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Paper: Makrai, L., Fodróczy, B., Nagy, S. Á., Czeiszing, P., Csabai, I., Szita, G. & Solymosi, N. (2023). Annotated dataset for deep-learning-based bacterial colony detection. *Scientific Data* 10, 497. <https://doi.org/10.1038/s41597-023-02404-8>

Changes made: the shipped COCO annotations (`annot_COCO.json`) are used; images are split 70/15/15 into train/val/test by species with a fixed seed.

### Clinical set (CC BY 4.0)

Dataset: Du, J. (2026). *Clinical Dataset of 19 Bacterial Colony Images*. Zenodo. <https://doi.org/10.5281/zenodo.18256864>. Licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

Paper: Du, J., Yang, C., Sun, M., Sun, Q., Wang, M., Ji, X., Wang, K. & Xu, J. (2026). High-Resolution Colony Images of Clinically Isolated Bacteria for Automated Detection and Deep Learning. *Scientific Data* 13, 757. <https://doi.org/10.1038/s41597-026-07095-5> (its reference list cites the Zenodo DOI above).

Changes made: labels converted to COCO format; images split 70/15/15 into train/val/test by isolate with a fixed seed; the augmented (flipped) `images_agu.zip` and `label_agu.zip` files are not used.

### AGAR (CC BY-NC 2.0)

Paper: Majchrowska, S., Pawłowski, J., Guła, G., Bonus, T., Hanas, A., Loch, A., Pawlak, A., Roszkowiak, J., Golan, T. & Drulis-Kawa, Z. (2021). AGAR a microbial colony dataset for deep learning detection. arXiv:2108.01234. <https://arxiv.org/abs/2108.01234>

Dataset: NeuroSYS Sp. z o.o., <https://agar.neurosys.com>. Licensed under [CC BY-NC 2.0](https://creativecommons.org/licenses/by-nc/2.0/): non-commercial use only.

Changes made: converted to COCO format (`train/prepare_agar.py`); colony boxes only (1,055 Defect/Contamination boxes dropped); the 4,511 uncountable plates (no boxes) are in no split and are listed in `splits.json`; the 13,489 countable and empty plates are split 70/15/15 (train 9,443 / val 2,023 / test 2,023) by plate group with a fixed seed, stratified by imaging setup (bright, dark, vague, lower resolution). A plate group is a set of byte-identical images (the release has 6 such pairs, 4 across its own train/validation/unassigned lists), so no plate is in two splits; the mirror's own lists are not used. Plates are resized per setup so the median colony box is about 64 px, like the public mix (factors bright 0.88, dark 0.6, lower resolution 0.74, vague 1.0; stored in the json `info`); the 640 px training tiles are cut from the resized plates. Output (outside git, under `COLONYCOUNT_AGAR_DIR`): `annotations/{train,val,test}.json`, `plates/{val,test}/` (resized plates, 3.7 GB) and `tiles_640/{train,valid}/` (88,641 + 18,571 tiles), 13.8 GB in total.

Mirror used for download: Hugging Face dataset [`leodriesch/agar`](https://huggingface.co/datasets/leodriesch/agar), pinned to revision `9ce24ca6611a700b2c61d22946786630c65e7764` (last modified 2026-10-15, card license `cc-by-nc-2.0`), 83 files, 33,502,602,770 bytes (33.5 GB). Fetched over plain HTTP from that pinned revision with the Python standard library (no `huggingface_hub`, no Xet; the Xet transfer ran at about 46 kB/s where plain HTTP gave 7-10 MB/s) and verified (SHA-256 per file) with `uv run --group prepare python train/prepare_agar.py`; the location is `COLONYCOUNT_AGAR_DIR` (default `<COLONYCOUNT_DATA_DIR>/agar`). Non-commercial use only, as above.

## Models and their training data

| Model | Trained on | Weights license | Status |
|---|---|---|---|
| `rfdetr-nano` (default) | smartphone + clinical | CC BY 4.0 (attribution above) | v0.1 |
| `yolox-tiny` | smartphone + clinical | CC BY 4.0 (attribution above) | v0.1 |
| `yolox-m` | smartphone + clinical | CC BY 4.0 (attribution above) | v0.1 |
| `agar` | AGAR | CC BY-NC 2.0, non-commercial only | later |

The code is Apache-2.0. Datasets and model weights are not covered by the code license; each carries the license listed above.
colonycount is free and open source, not a commercial product.
