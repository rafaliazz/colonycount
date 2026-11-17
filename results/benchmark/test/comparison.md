| method | setting | threshold | CPU s/plate (median) |
|---|---|---|---|
| RF-DETR-nano (crop) | public-rfdetr-nano-20261113.onnx | 0.4 | 8.98 |
| YOLOv8-n (crop) | yolov8n-20261113.onnx | 0.4 | 1.1 |
| YOLOv8-m (crop) | yolov8m-20261113.onnx | 0.35 | 6.63 |
| YOLOX-Tiny (crop) | yolox_tiny-20261114.onnx | 0.5 | 1.6 |
| YOLOX-M (crop) | yolox_m-20261114.onnx | 0.35 | 5.95 |
| OpenCFU (crop) | -d bil -t 5 -r 4 -R 30 | - | 1.05 |
| RF-DETR-nano (whole image) | public-rfdetr-nano-20261113.onnx | 0.35 | 17.41 |
| YOLOv8-n (whole image) | yolov8n-20261113.onnx | 0.35 | 1.83 |
| YOLOv8-m (whole image) | yolov8m-20261113.onnx | 0.35 | 11.01 |
| YOLOX-Tiny (whole image) | yolox_tiny-20261114.onnx | 0.5 | 2.64 |
| YOLOX-M (whole image) | yolox_m-20261114.onnx | 0.45 | 9.95 |
| OpenCFU (whole image) | -d bil -t 5 -r 6 -R 45 | - | 1.02 |

| clinical | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 142 | 0.739 | 0.077 | 0.567 | 70 | 0.9905 | 0.9919 | 1.0 |
| YOLOv8-n (crop) | 142 | 0.958 | 0.352 | 0.986 | 70 | 0.9862 | 0.9927 | 1.0 |
| YOLOv8-m (crop) | 142 | 0.979 | 0.387 | 0.986 | 70 | 0.9856 | 0.9927 | 1.0 |
| YOLOX-Tiny (crop) | 142 | 0.711 | 0.275 | 0.457 | 70 | 0.9891 | 0.9941 | 1.0 |
| YOLOX-M (crop) | 142 | 0.683 | 0.218 | 0.58 | 70 | 0.9901 | 0.9941 | 1.0 |
| OpenCFU (crop) | 142 | 8.831 | 3.014 | 9.726 | 70 | 0.8418 | 0.889 | 1.0 |
| RF-DETR-nano (whole image) | 142 | 0.859 | 0.31 | 1.156 | 70 | 0.9869 | 0.9925 | 0.0 |
| YOLOv8-n (whole image) | 142 | 0.993 | 0.486 | 0.937 | 70 | 0.9844 | 0.9933 | 0.0 |
| YOLOv8-m (whole image) | 142 | 1.042 | 0.423 | 1.227 | 70 | 0.9851 | 0.9928 | 0.0 |
| YOLOX-Tiny (whole image) | 142 | 0.732 | 0.324 | 0.803 | 70 | 0.9885 | 0.9945 | 0.0 |
| YOLOX-M (whole image) | 142 | 0.789 | 0.254 | 0.612 | 70 | 0.9894 | 0.9941 | 0.0 |
| OpenCFU (whole image) | 142 | 5.62 | -0.69 | 6.581 | 70 | 0.9199 | 0.9081 | 0.0 |

| smartphone | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 50 | 1.96 | -0.28 | 1.852 | 44 | 0.9709 | 0.9682 | 1.0 |
| YOLOv8-n (crop) | 50 | 6.38 | -6.06 | 5.719 | 44 | 0.9777 | 0.9196 | 1.0 |
| YOLOv8-m (crop) | 50 | 5.84 | -5.4 | 5.132 | 44 | 0.9799 | 0.928 | 1.0 |
| YOLOX-Tiny (crop) | 50 | 5.06 | -4.54 | 4.42 | 44 | 0.9793 | 0.9357 | 1.0 |
| YOLOX-M (crop) | 50 | 2.04 | -0.56 | 1.44 | 44 | 0.9673 | 0.962 | 1.0 |
| OpenCFU (crop) | 50 | 15.7 | 7.38 | 13.295 | 44 | 0.8054 | 0.8637 | 1.0 |
| RF-DETR-nano (whole image) | 50 | 2.18 | 1.34 | 1.81 | 44 | 0.963 | 0.9757 | 0.0 |
| YOLOv8-n (whole image) | 50 | 4.8 | -4.2 | 4.557 | 44 | 0.9724 | 0.9323 | 0.0 |
| YOLOv8-m (whole image) | 50 | 5.9 | -5.5 | 5.548 | 44 | 0.9803 | 0.9274 | 0.0 |
| YOLOX-Tiny (whole image) | 50 | 5.08 | -4.36 | 4.42 | 44 | 0.9777 | 0.9359 | 0.0 |
| YOLOX-M (whole image) | 50 | 3.26 | -2.74 | 2.538 | 44 | 0.9736 | 0.9474 | 0.0 |
| OpenCFU (whole image) | 50 | 21.9 | 20.54 | 18.788 | 44 | 0.7507 | 0.9019 | 0.0 |

| overall | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 192 | 1.057 | -0.016 | 1.061 | 114 | 0.9826 | 0.9824 | 1.0 |
| YOLOv8-n (crop) | 192 | 2.37 | -1.318 | 2.107 | 114 | 0.9829 | 0.9634 | 1.0 |
| YOLOv8-m (crop) | 192 | 2.245 | -1.12 | 1.527 | 114 | 0.9834 | 0.9668 | 1.0 |
| YOLOX-Tiny (crop) | 192 | 1.844 | -0.979 | 1.449 | 114 | 0.9853 | 0.9707 | 1.0 |
| YOLOX-M (crop) | 192 | 1.036 | 0.016 | 0.924 | 114 | 0.981 | 0.9812 | 1.0 |
| OpenCFU (crop) | 192 | 10.62 | 4.151 | 10.118 | 114 | 0.8271 | 0.8789 | 1.0 |
| RF-DETR-nano (whole image) | 192 | 1.203 | 0.578 | 1.449 | 114 | 0.9773 | 0.9858 | 0.0 |
| YOLOv8-n (whole image) | 192 | 1.984 | -0.734 | 1.653 | 114 | 0.9798 | 0.9689 | 0.0 |
| YOLOv8-m (whole image) | 192 | 2.307 | -1.12 | 1.607 | 114 | 0.9832 | 0.9666 | 0.0 |
| YOLOX-Tiny (whole image) | 192 | 1.865 | -0.896 | 1.589 | 114 | 0.9843 | 0.971 | 0.0 |
| YOLOX-M (whole image) | 192 | 1.432 | -0.526 | 1.156 | 114 | 0.9832 | 0.9754 | 0.0 |
| OpenCFU (whole image) | 192 | 9.859 | 4.839 | 7.953 | 114 | 0.8441 | 0.9056 | 0.0 |

| clinical, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 72 | 0.486 | 0.486 | - | 0 | 0.9666 | 1.0 | 1.0 |
| YOLOv8-n (crop) | 72 | 0.722 | 0.694 | - | 0 | 0.9521 | 0.999 | 1.0 |
| YOLOv8-m (crop) | 72 | 0.694 | 0.694 | - | 0 | 0.953 | 1.0 | 1.0 |
| YOLOX-Tiny (crop) | 72 | 0.5 | 0.5 | - | 0 | 0.9657 | 1.0 | 1.0 |
| YOLOX-M (crop) | 72 | 0.417 | 0.417 | - | 0 | 0.9713 | 1.0 | 1.0 |
| OpenCFU (crop) | 72 | 7.403 | 7.319 | - | 0 | 0.5983 | 0.9093 | 1.0 |
| RF-DETR-nano (whole image) | 72 | 0.639 | 0.639 | - | 0 | 0.9566 | 1.0 | 0.0 |
| YOLOv8-n (whole image) | 72 | 0.792 | 0.792 | - | 0 | 0.9468 | 1.0 | 0.0 |
| YOLOv8-m (whole image) | 72 | 0.75 | 0.75 | - | 0 | 0.9494 | 1.0 | 0.0 |
| YOLOX-Tiny (whole image) | 72 | 0.542 | 0.542 | - | 0 | 0.963 | 1.0 | 0.0 |
| YOLOX-M (whole image) | 72 | 0.528 | 0.528 | - | 0 | 0.9639 | 1.0 | 0.0 |
| OpenCFU (whole image) | 72 | 4.125 | 3.792 | - | 0 | 0.749 | 0.9507 | 0.0 |

| clinical, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 70 | 1.0 | -0.343 | 0.567 | 70 | 0.9942 | 0.9906 | 1.0 |
| YOLOv8-n (crop) | 70 | 1.2 | 0.0 | 0.986 | 70 | 0.9917 | 0.9917 | 1.0 |
| YOLOv8-m (crop) | 70 | 1.271 | 0.071 | 0.986 | 70 | 0.9908 | 0.9915 | 1.0 |
| YOLOX-Tiny (crop) | 70 | 0.929 | 0.043 | 0.457 | 70 | 0.9928 | 0.9932 | 1.0 |
| YOLOX-M (crop) | 70 | 0.957 | 0.014 | 0.58 | 70 | 0.9931 | 0.9932 | 1.0 |
| OpenCFU (crop) | 70 | 10.3 | -1.414 | 9.726 | 70 | 0.8993 | 0.8859 | 1.0 |
| RF-DETR-nano (whole image) | 70 | 1.086 | -0.029 | 1.156 | 70 | 0.9917 | 0.9914 | 0.0 |
| YOLOv8-n (whole image) | 70 | 1.2 | 0.171 | 0.937 | 70 | 0.9905 | 0.9923 | 0.0 |
| YOLOv8-m (whole image) | 70 | 1.343 | 0.086 | 1.227 | 70 | 0.9908 | 0.9917 | 0.0 |
| YOLOX-Tiny (whole image) | 70 | 0.929 | 0.1 | 0.803 | 70 | 0.9926 | 0.9937 | 0.0 |
| YOLOX-M (whole image) | 70 | 1.057 | -0.029 | 0.612 | 70 | 0.9935 | 0.9932 | 0.0 |
| OpenCFU (whole image) | 70 | 7.157 | -5.3 | 6.581 | 70 | 0.9551 | 0.9016 | 0.0 |

| clinical, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (crop) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (crop) | 0 | - | - | - | 0 | - | - | - |
| RF-DETR-nano (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (whole image) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (whole image) | 0 | - | - | - | 0 | - | - | - |

| smartphone, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 6 | 0.5 | -0.167 | - | 0 | 0.9833 | 0.9672 | 1.0 |
| YOLOv8-n (crop) | 6 | 2.333 | -2.333 | - | 0 | 0.9787 | 0.7541 | 1.0 |
| YOLOv8-m (crop) | 6 | 1.167 | -1.167 | - | 0 | 0.9815 | 0.8689 | 1.0 |
| YOLOX-Tiny (crop) | 6 | 0.833 | -0.833 | - | 0 | 1.0 | 0.918 | 1.0 |
| YOLOX-M (crop) | 6 | 1.0 | 0.0 | - | 0 | 0.9508 | 0.9508 | 1.0 |
| OpenCFU (crop) | 6 | 2.333 | 0.667 | - | 0 | 0.7692 | 0.8197 | 1.0 |
| RF-DETR-nano (whole image) | 6 | 0.333 | 0.0 | - | 0 | 0.9836 | 0.9836 | 0.0 |
| YOLOv8-n (whole image) | 6 | 0.667 | -0.667 | - | 0 | 0.9825 | 0.918 | 0.0 |
| YOLOv8-m (whole image) | 6 | 1.167 | -1.167 | - | 0 | 0.9815 | 0.8689 | 0.0 |
| YOLOX-Tiny (whole image) | 6 | 1.0 | -1.0 | - | 0 | 1.0 | 0.9016 | 0.0 |
| YOLOX-M (whole image) | 6 | 1.0 | 0.0 | - | 0 | 0.9508 | 0.9508 | 0.0 |
| OpenCFU (whole image) | 6 | 4.333 | 4.333 | - | 0 | 0.6322 | 0.9016 | 0.0 |

| smartphone, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 44 | 2.159 | -0.295 | 1.852 | 44 | 0.9707 | 0.9682 | 1.0 |
| YOLOv8-n (crop) | 44 | 6.932 | -6.568 | 5.719 | 44 | 0.9777 | 0.9216 | 1.0 |
| YOLOv8-m (crop) | 44 | 6.477 | -5.977 | 5.132 | 44 | 0.9799 | 0.9287 | 1.0 |
| YOLOX-Tiny (crop) | 44 | 5.636 | -5.045 | 4.42 | 44 | 0.979 | 0.9359 | 1.0 |
| YOLOX-M (crop) | 44 | 2.182 | -0.636 | 1.44 | 44 | 0.9675 | 0.9621 | 1.0 |
| OpenCFU (crop) | 44 | 17.523 | 8.295 | 13.295 | 44 | 0.8058 | 0.8642 | 1.0 |
| RF-DETR-nano (whole image) | 44 | 2.432 | 1.523 | 1.81 | 44 | 0.9628 | 0.9756 | 0.0 |
| YOLOv8-n (whole image) | 44 | 5.364 | -4.682 | 4.557 | 44 | 0.9723 | 0.9325 | 0.0 |
| YOLOv8-m (whole image) | 44 | 6.545 | -6.091 | 5.548 | 44 | 0.9803 | 0.9281 | 0.0 |
| YOLOX-Tiny (whole image) | 44 | 5.636 | -4.818 | 4.42 | 44 | 0.9774 | 0.9363 | 0.0 |
| YOLOX-M (whole image) | 44 | 3.568 | -3.114 | 2.538 | 44 | 0.9739 | 0.9474 | 0.0 |
| OpenCFU (whole image) | 44 | 24.295 | 22.75 | 18.788 | 44 | 0.7524 | 0.9019 | 0.0 |

| smartphone, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (crop) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (crop) | 0 | - | - | - | 0 | - | - | - |
| RF-DETR-nano (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (whole image) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (whole image) | 0 | - | - | - | 0 | - | - | - |

| all, GT <30 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 78 | 0.487 | 0.436 | - | 0 | 0.9675 | 0.9981 | 1.0 |
| YOLOv8-n (crop) | 78 | 0.846 | 0.462 | - | 0 | 0.9532 | 0.9851 | 1.0 |
| YOLOv8-m (crop) | 78 | 0.731 | 0.551 | - | 0 | 0.9544 | 0.9926 | 1.0 |
| YOLOX-Tiny (crop) | 78 | 0.526 | 0.397 | - | 0 | 0.9675 | 0.9953 | 1.0 |
| YOLOX-M (crop) | 78 | 0.462 | 0.385 | - | 0 | 0.9701 | 0.9972 | 1.0 |
| OpenCFU (crop) | 78 | 7.013 | 6.808 | - | 0 | 0.6052 | 0.9042 | 1.0 |
| RF-DETR-nano (whole image) | 78 | 0.615 | 0.59 | - | 0 | 0.9581 | 0.9991 | 0.0 |
| YOLOv8-n (whole image) | 78 | 0.782 | 0.679 | - | 0 | 0.9486 | 0.9953 | 0.0 |
| YOLOv8-m (whole image) | 78 | 0.782 | 0.603 | - | 0 | 0.951 | 0.9926 | 0.0 |
| YOLOX-Tiny (whole image) | 78 | 0.577 | 0.423 | - | 0 | 0.9648 | 0.9944 | 0.0 |
| YOLOX-M (whole image) | 78 | 0.564 | 0.487 | - | 0 | 0.9632 | 0.9972 | 0.0 |
| OpenCFU (whole image) | 78 | 4.141 | 3.833 | - | 0 | 0.7416 | 0.9479 | 0.0 |

| all, GT 30-300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 114 | 1.447 | -0.325 | 1.061 | 114 | 0.9841 | 0.981 | 1.0 |
| YOLOv8-n (crop) | 114 | 3.412 | -2.535 | 2.107 | 114 | 0.9858 | 0.9614 | 1.0 |
| YOLOv8-m (crop) | 114 | 3.281 | -2.263 | 1.527 | 114 | 0.9862 | 0.9644 | 1.0 |
| YOLOX-Tiny (crop) | 114 | 2.746 | -1.921 | 1.449 | 114 | 0.987 | 0.9684 | 1.0 |
| YOLOX-M (crop) | 114 | 1.43 | -0.237 | 0.924 | 114 | 0.982 | 0.9798 | 1.0 |
| OpenCFU (crop) | 114 | 13.088 | 2.333 | 10.118 | 114 | 0.857 | 0.8765 | 1.0 |
| RF-DETR-nano (whole image) | 114 | 1.605 | 0.57 | 1.449 | 114 | 0.9791 | 0.9846 | 0.0 |
| YOLOv8-n (whole image) | 114 | 2.807 | -1.702 | 1.653 | 114 | 0.9828 | 0.9665 | 0.0 |
| YOLOv8-m (whole image) | 114 | 3.351 | -2.298 | 1.607 | 114 | 0.9864 | 0.9642 | 0.0 |
| YOLOX-Tiny (whole image) | 114 | 2.746 | -1.798 | 1.589 | 114 | 0.9862 | 0.9689 | 0.0 |
| YOLOX-M (whole image) | 114 | 2.026 | -1.219 | 1.156 | 114 | 0.9852 | 0.9734 | 0.0 |
| OpenCFU (whole image) | 114 | 13.772 | 5.526 | 7.953 | 114 | 0.8555 | 0.9017 | 0.0 |

| all, GT >300 | n | mae | bias | mdape_30_300 | n_30_300 | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|
| RF-DETR-nano (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (crop) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (crop) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (crop) | 0 | - | - | - | 0 | - | - | - |
| RF-DETR-nano (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-n (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOv8-m (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-Tiny (whole image) | 0 | - | - | - | 0 | - | - | - |
| YOLOX-M (whole image) | 0 | - | - | - | 0 | - | - | - |
| OpenCFU (whole image) | 0 | - | - | - | 0 | - | - | - |

