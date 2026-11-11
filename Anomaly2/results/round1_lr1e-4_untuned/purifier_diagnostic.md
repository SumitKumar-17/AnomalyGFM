### Purifier edge-level diagnostic (5 seeds)
|                                 |   ('edge_auroc', 'mean') |   ('edge_auroc', 'std') |   ('drop_injected', 'mean') |   ('drop_injected', 'std') |   ('drop_original', 'mean') |   ('drop_original', 'std') |   ('model_auroc_attacked', 'mean') |   ('model_auroc_attacked', 'std') |
|:--------------------------------|-------------------------:|------------------------:|----------------------------:|---------------------------:|----------------------------:|---------------------------:|-----------------------------------:|----------------------------------:|
| ('amazon_upu', 'camouflage 20') |                    0.152 |                   0.034 |                           0 |                          0 |                           0 |                          0 |                              0.186 |                             0.178 |
| ('reddit', 'camouflage 20')     |                    0.161 |                   0.169 |                           0 |                          0 |                           0 |                          0 |                              0.182 |                             0.034 |
| ('amazon_upu', 'rewire 0.3')    |                    0.117 |                   0.019 |                           0 |                          0 |                           0 |                          0 |                              0.664 |                             0.132 |

### per seed
| target     | attack        |   seed |   n_injected |   n_edges |   edge_auroc |   drop_injected |   drop_original |   model_auroc_attacked |
|:-----------|:--------------|-------:|-------------:|----------:|-------------:|----------------:|----------------:|-----------------------:|
| amazon_upu | camouflage 20 |      0 |        13773 |    189381 |        0.128 |               0 |               0 |                  0.101 |
| amazon_upu | camouflage 20 |      1 |        13769 |    189377 |        0.166 |               0 |               0 |                  0.5   |
| amazon_upu | camouflage 20 |      2 |        13783 |    189391 |        0.207 |               0 |               0 |                  0.074 |
| amazon_upu | camouflage 20 |      3 |        13783 |    189391 |        0.129 |               0 |               0 |                  0.157 |
| amazon_upu | camouflage 20 |      4 |        13778 |    189386 |        0.132 |               0 |               0 |                  0.097 |
| reddit     | camouflage 20 |      0 |         7309 |     85825 |        0.116 |               0 |               0 |                  0.214 |
| reddit     | camouflage 20 |      1 |         7303 |     85819 |        0.075 |               0 |               0 |                  0.191 |
| reddit     | camouflage 20 |      2 |         7309 |     85825 |        0.072 |               0 |               0 |                  0.204 |
| reddit     | camouflage 20 |      3 |         7307 |     85823 |        0.461 |               0 |               0 |                  0.126 |
| reddit     | camouflage 20 |      4 |         7307 |     85823 |        0.081 |               0 |               0 |                  0.175 |
| amazon_upu | rewire 0.3    |      0 |        52478 |    228086 |        0.095 |               0 |               0 |                  0.833 |
| amazon_upu | rewire 0.3    |      1 |        52460 |    228068 |        0.112 |               0 |               0 |                  0.727 |
| amazon_upu | rewire 0.3    |      2 |        52473 |    228081 |        0.146 |               0 |               0 |                  0.506 |
| amazon_upu | rewire 0.3    |      3 |        52466 |    228074 |        0.12  |               0 |               0 |                  0.559 |
| amazon_upu | rewire 0.3    |      4 |        52483 |    228091 |        0.112 |               0 |               0 |                  0.696 |