# Changelog

## 0.1.0 - 2026-11-22

First release.

- Three built-in detectors, downloaded on first use from Hugging Face (`rafai-azz/colonycount-public`, CC BY 4.0 weights): `rfdetr-nano` (default), `yolox-tiny` (fastest) and `yolox-m`. Test-split MdAPE on plates with 30-300 colonies: 1.06%, 1.45% and 0.92%.
- Python API (`colonycount.count`) and the `colonycount` CLI, with CSV output, CFU/mL and TFTC/TNTC range flags.
- Plate finder: counting is restricted to the plate rim by default.
- Benchmark against OpenCFU (10.12% MdAPE) and YOLOv8, committed under [results/](results/README.md).
- Code is Apache-2.0; dataset licenses and attribution are in [DATA.md](DATA.md).
