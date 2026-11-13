# OpenCFU baseline

[OpenCFU](https://opencfu.sourceforge.net) 4.0.0 (Debian package `opencfu 4.0.0+dfsg-2+b2`; its
`-v` prints 3.9.1) in a pinned `debian:stable-slim` image, run as a non-root user without network.

```sh
docker build -t colonycount-opencfu bench/opencfu
uv run python bench/opencfu_runner.py --tune               # val, plate crop; grid-search settings
uv run python bench/opencfu_runner.py --tune --no-plate    # val, whole image
```

The image's working directory is `/usr/share/opencfu`, because `opencfu` loads its colony
classifier from `./data/trainedClassifier*.xml`.

## Command line used

`opencfu -i FILE [flags]` prints one CSV row per object on stdout:
`IsValid,X,Y,ROI,Colour_group,N_in_clust,Area,Radius,Hue,Saturation,Rmean,...`. Rows with
`IsValid=1` are counted; `X`, `Y` (centre) and `Radius` are in pixels of the input image, so
per-colony precision/recall can be scored.

| flag | meaning | used |
|---|---|---|
| `-d {reg,inv,bil}` | threshold type: regular, inverted, bilateral (default `reg`) | tuned |
| `-t NUM` | threshold value (default 10) | tuned |
| `-a` | automatic threshold; overrides `-t` | tuned (`auto`) |
| `-r NUM` / `-R NUM` | minimum / maximum colony radius in pixels | tuned, as a fraction of the image side |
| `-m auto` | automatic plate mask | not used: it returned 0 colonies on every val plate tried |
| `-l`, `-c/-C`, `-D/-N` | outlier, hue and colour-clustering filters | defaults |

Photos are resized (longest side 1000 or 1500 px, tuned) before OpenCFU sees them: at full
resolution (12 MP clinical photos) it took ~25 s per plate, found nothing with default radii and
ran out of memory with `-a`.
