## OpenCFU 4.0.0 (-d bil -t 5 -r 6 -R 45, longest side 1500 px) on test (plate restriction off)

| source | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| clinical | 142 | 5.62 | -0.69 | 6.581 | 70 | 9.233 | 0.9199 | 0.9081 | 0.0 |
| smartphone | 50 | 21.9 | 20.54 | 18.788 | 44 | 19.371 | 0.7507 | 0.9019 | 0.0 |
| overall | 192 | 9.859 | 4.839 | 7.953 | 114 | 11.028 | 0.8441 | 0.9056 | 0.0 |

| clinical by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 72 | 4.125 | 3.792 | - | 0 | 20.345 | 0.749 | 0.9507 | 0.0 |
| 30-300 | 70 | 7.157 | -5.3 | 6.581 | 70 | 6.581 | 0.9551 | 0.9016 | 0.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

| smartphone by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 6 | 4.333 | 4.333 | - | 0 | 36.905 | 0.6322 | 0.9016 | 0.0 |
| 30-300 | 44 | 24.295 | 22.75 | 18.788 | 44 | 18.788 | 0.7524 | 0.9019 | 0.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

| all by GT count | n | mae | bias | mdape_30_300 | n_30_300 | mdape_all | precision | recall | plate_found |
|---|---|---|---|---|---|---|---|---|---|
| <30 | 78 | 4.141 | 3.833 | - | 0 | 21.059 | 0.7416 | 0.9479 | 0.0 |
| 30-300 | 114 | 13.772 | 5.526 | 7.953 | 114 | 7.953 | 0.8555 | 0.9017 | 0.0 |
| >300 | 0 | - | - | - | 0 | - | - | - | - |

plate not found: 192 of 192 plates; GT colonies outside the found plate circle: {'clinical': 0.0, 'smartphone': 0.0, 'all': 0.0}
CPU time per plate: median 1.02 s, mean 1.03 s, max 2.01 s
