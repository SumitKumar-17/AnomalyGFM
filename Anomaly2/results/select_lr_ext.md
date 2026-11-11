### Source-only selection of lr_ext (Facebook held-out nodes; 3 seeds)
|                                  |   fb_heldout_auroc |   edge_auroc |   drop_injected |   drop_original |
|:---------------------------------|-------------------:|-------------:|----------------:|----------------:|
| ('MoE K=4', 0.0001)              |              0.808 |      nan     |         nan     |         nan     |
| ('MoE K=4', 0.001)               |              0.82  |      nan     |         nan     |         nan     |
| ('MoE K=4', 0.01)                |              0.753 |      nan     |         nan     |         nan     |
| ('MoE K=4', 0.03)                |              0.64  |      nan     |         nan     |         nan     |
| ('Purifier', 0.0001)             |              0.546 |        0.297 |           0.003 |           0.002 |
| ('Purifier', 0.001)              |              0.473 |        0.438 |           0.57  |           0.596 |
| ('Purifier', 0.01)               |              0.392 |        0.699 |           0.73  |           0.484 |
| ('Purifier', 0.03)               |              0.411 |        0.708 |           0.725 |           0.456 |
| ('Adaptive PPR', 0.0001)         |              0.787 |      nan     |         nan     |         nan     |
| ('Adaptive PPR', 0.001)          |              0.788 |      nan     |         nan     |         nan     |
| ('Adaptive PPR', 0.01)           |              0.828 |      nan     |         nan     |         nan     |
| ('Adaptive PPR', 0.03)           |              0.825 |      nan     |         nan     |         nan     |
| ('Baseline (reference)', 0.0001) |              0.828 |      nan     |         nan     |         nan     |
| ('Baseline (reference)', 0.001)  |              0.828 |      nan     |         nan     |         nan     |
| ('Baseline (reference)', 0.01)   |              0.828 |      nan     |         nan     |         nan     |
| ('Baseline (reference)', 0.03)   |              0.828 |      nan     |         nan     |         nan     |