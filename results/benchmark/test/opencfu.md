## OpenCFU 4.0.0 (-d bil -t 5 -r 4 -R 30, longest side 1000 px) on test (plate restriction on)

| source | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| clinical | 142 | 8.831 | 3.014 | 9.726 | 70 | 15.301 | 0.8418 | 0.889 | 1.0 |
| smartphone | 50 | 15.7 | 7.38 | 13.295 | 44 | 13.295 | 0.8054 | 0.8637 | 1.0 |
| overall | 192 | 10.62 | 4.151 | 10.118 | 114 | 14.327 | 0.8271 | 0.8789 | 1.0 |

| clinical by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 72 | 7.403 | 7.319 | - | 0 | 32.456 | 0.5983 | 0.9093 | 1.0 |
| 30-300 | 70 | 10.3 | -1.414 | 9.726 | 70 | 9.726 | 0.8993 | 0.8859 | 1.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

| smartphone by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 6 | 2.333 | 0.667 | - | 0 | 11.905 | 0.7692 | 0.8197 | 1.0 |
| 30-300 | 44 | 17.523 | 8.295 | 13.295 | 44 | 13.295 | 0.8058 | 0.8642 | 1.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

| all by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 78 | 7.013 | 6.808 | - | 0 | 30.075 | 0.6052 | 0.9042 | 1.0 |
| 30-300 | 114 | 13.088 | 2.333 | 10.118 | 114 | 10.118 | 0.857 | 0.8765 | 1.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

plate not found: 0 of 192 plates; GT colonies outside the found plate circle: {'clinical': 0.0001, 'smartphone': 0.0008, 'all': 0.0004}
CPU time per plate: median 1.05 s, mean 0.95 s, max 1.77 s
