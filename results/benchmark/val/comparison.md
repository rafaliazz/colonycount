| method | setting | threshold | CPU s/plate (median) |
|---|---|---|---|
| RF-DETR-nano (crop) | public-rfdetr-nano-20261113.onnx | 0.45 | 14.35 |
| OpenCFU (crop) | -d bil -t 5 -r 4 -R 60 | - | 0.56 |
| RF-DETR-nano (whole image) | public-rfdetr-nano-20261113.onnx | 0.45 | 24.46 |
| OpenCFU (whole image) | -d bil -t 3 -r 4 -R 30 | - | 0.49 |

| clinical | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 142 | 3.641 | 0.824 | 2.542 | 100 | 0.9612 | 0.973 | 0.6127 |
| OpenCFU (crop) | 142 | 12.641 | 4.289 | 9.623 | 100 | 0.7562 | 0.8046 | 0.6127 |
| RF-DETR-nano (whole image) | 142 | 3.218 | 2.838 | 1.653 | 100 | 0.9513 | 0.9916 | 0.0 |
| OpenCFU (whole image) | 142 | 12.592 | -0.113 | 8.819 | 100 | 0.8211 | 0.8197 | 0.0 |

| smartphone | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 56 | 11.161 | -6.196 | 1.702 | 40 | 0.9709 | 0.9347 | 1.0 |
| OpenCFU (crop) | 56 | 50.071 | -44.321 | 11.07 | 40 | 0.9281 | 0.6807 | 1.0 |
| RF-DETR-nano (whole image) | 56 | 9.589 | -5.054 | 1.262 | 40 | 0.9731 | 0.9435 | 0.0 |
| OpenCFU (whole image) | 56 | 48.464 | -34.357 | 7.634 | 40 | 0.881 | 0.699 | 0.0 |

| overall | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 198 | 5.768 | -1.162 | 2.204 | 140 | 0.9658 | 0.954 | 0.7222 |
| OpenCFU (crop) | 198 | 23.227 | -9.46 | 9.989 | 140 | 0.8255 | 0.7433 | 0.7222 |
| RF-DETR-nano (whole image) | 198 | 5.02 | 0.606 | 1.429 | 140 | 0.9617 | 0.9678 | 0.0 |
| OpenCFU (whole image) | 198 | 22.737 | -9.798 | 8.661 | 140 | 0.8473 | 0.76 | 0.0 |

| clinical, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 40 | 2.025 | 1.225 | - | 0 | 0.9128 | 0.974 | 0.55 |
| OpenCFU (crop) | 40 | 10.25 | 8.25 | - | 0 | 0.4863 | 0.7059 | 0.55 |
| RF-DETR-nano (whole image) | 40 | 2.55 | 2.55 | - | 0 | 0.8715 | 0.9932 | 0.0 |
| OpenCFU (whole image) | 40 | 8.475 | 8.425 | - | 0 | 0.5927 | 0.8659 | 0.0 |

| clinical, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 100 | 4.18 | 0.82 | 2.542 | 100 | 0.9635 | 0.9734 | 0.63 |
| OpenCFU (crop) | 100 | 12.51 | 4.13 | 9.623 | 100 | 0.7771 | 0.817 | 0.63 |
| RF-DETR-nano (whole image) | 100 | 3.5 | 2.98 | 1.653 | 100 | 0.9562 | 0.9917 | 0.0 |
| OpenCFU (whole image) | 100 | 11.8 | -0.84 | 8.819 | 100 | 0.8421 | 0.8333 | 0.0 |

| clinical, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 2 | 9.0 | -7.0 | - | 0 | 0.9864 | 0.9679 | 1.0 |
| OpenCFU (crop) | 2 | 67.0 | -67.0 | - | 0 | 0.9364 | 0.7684 | 1.0 |
| RF-DETR-nano (whole image) | 2 | 2.5 | 1.5 | - | 0 | 0.9853 | 0.9893 | 0.0 |
| OpenCFU (whole image) | 2 | 134.5 | -134.5 | - | 0 | 0.9812 | 0.6278 | 0.0 |

| smartphone, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 6 | 2.333 | 1.0 | - | 0 | 0.6471 | 0.7857 | 1.0 |
| OpenCFU (crop) | 6 | 7.5 | 4.833 | - | 0 | 0.2281 | 0.4643 | 1.0 |
| RF-DETR-nano (whole image) | 6 | 1.667 | 0.0 | - | 0 | 0.75 | 0.75 | 0.0 |
| OpenCFU (whole image) | 6 | 20.833 | 18.167 | - | 0 | 0.0876 | 0.4286 | 0.0 |

| smartphone, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 40 | 7.15 | -3.8 | 1.702 | 40 | 0.9759 | 0.9472 | 1.0 |
| OpenCFU (crop) | 40 | 31.425 | -25.225 | 11.07 | 40 | 0.9246 | 0.7442 | 1.0 |
| RF-DETR-nano (whole image) | 40 | 5.975 | -2.875 | 1.262 | 40 | 0.9777 | 0.9559 | 0.0 |
| OpenCFU (whole image) | 40 | 29.575 | -15.675 | 7.634 | 40 | 0.8805 | 0.7738 | 0.0 |

| smartphone, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 10 | 32.5 | -20.1 | - | 0 | 0.9673 | 0.92 | 1.0 |
| OpenCFU (crop) | 10 | 150.2 | -150.2 | - | 0 | 0.949 | 0.6024 | 1.0 |
| RF-DETR-nano (whole image) | 10 | 28.8 | -16.8 | - | 0 | 0.9688 | 0.9292 | 0.0 |
| OpenCFU (whole image) | 10 | 140.6 | -140.6 | - | 0 | 0.922 | 0.6068 | 0.0 |

| all, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 46 | 2.065 | 1.196 | - | 0 | 0.9017 | 0.9671 | 0.6087 |
| OpenCFU (crop) | 46 | 9.891 | 7.804 | - | 0 | 0.4732 | 0.697 | 0.6087 |
| RF-DETR-nano (whole image) | 46 | 2.435 | 2.217 | - | 0 | 0.8676 | 0.9842 | 0.0 |
| OpenCFU (whole image) | 46 | 10.087 | 9.696 | - | 0 | 0.5353 | 0.8498 | 0.0 |

| all, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 140 | 5.029 | -0.5 | 2.204 | 140 | 0.9683 | 0.9631 | 0.7357 |
| OpenCFU (crop) | 140 | 17.914 | -4.257 | 9.989 | 140 | 0.8258 | 0.7885 | 0.7357 |
| RF-DETR-nano (whole image) | 140 | 4.207 | 1.307 | 1.429 | 140 | 0.9643 | 0.9777 | 0.0 |
| OpenCFU (whole image) | 140 | 16.879 | -5.079 | 8.661 | 140 | 0.8561 | 0.81 | 0.0 |

| all, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 12 | 28.583 | -17.917 | - | 0 | 0.9703 | 0.9274 | 1.0 |
| OpenCFU (crop) | 12 | 136.333 | -136.333 | - | 0 | 0.9466 | 0.6279 | 1.0 |
| RF-DETR-nano (whole image) | 12 | 24.417 | -13.75 | - | 0 | 0.9715 | 0.9385 | 0.0 |
| OpenCFU (whole image) | 12 | 139.583 | -139.583 | - | 0 | 0.9309 | 0.61 | 0.0 |

