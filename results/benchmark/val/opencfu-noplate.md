## OpenCFU 4.0.0 (-d bil -t 3 -r 4 -R 30, longest side 1000 px) on val (plate restriction off)

| source | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| clinical | 142 | 12.592 | -0.113 | 8.819 | 100 | 11.111 | 0.8211 | 0.8197 | 0.0 |
| smartphone | 56 | 48.464 | -34.357 | 7.634 | 40 | 14.462 | 0.881 | 0.699 | 0.0 |
| overall | 198 | 22.737 | -9.798 | 8.661 | 140 | 11.835 | 0.8473 | 0.76 | 0.0 |

| clinical by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 40 | 8.475 | 8.425 | - | 0 | 18.182 | 0.5927 | 0.8659 | 0.0 |
| 30-300 | 100 | 11.8 | -0.84 | 8.819 | 100 | 8.819 | 0.8421 | 0.8333 | 0.0 |
| >300 | 2 | 134.5 | -134.5 | - | 0 | 35.088 | 0.9812 | 0.6278 | 0.0 |

| smartphone by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 6 | 20.833 | 18.167 | - | 0 | 416.667 | 0.0876 | 0.4286 | 0.0 |
| 30-300 | 40 | 29.575 | -15.675 | 7.634 | 40 | 7.634 | 0.8805 | 0.7738 | 0.0 |
| >300 | 10 | 140.6 | -140.6 | - | 0 | 16.229 | 0.922 | 0.6068 | 0.0 |

| all by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 46 | 10.087 | 9.696 | - | 0 | 22.845 | 0.5353 | 0.8498 | 0.0 |
| 30-300 | 140 | 16.879 | -5.079 | 8.661 | 140 | 8.661 | 0.8561 | 0.81 | 0.0 |
| >300 | 12 | 139.583 | -139.583 | - | 0 | 21.599 | 0.9309 | 0.61 | 0.0 |

plate not found: 198 of 198 plates; GT colonies outside the found plate circle: {'clinical': 0.0, 'smartphone': 0.0, 'all': 0.0}
CPU time per plate: median 0.49 s, mean 0.54 s, max 1.14 s
