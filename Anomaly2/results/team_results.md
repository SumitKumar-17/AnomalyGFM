
### amazon  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.834±0.022 | 0.475±0.121 |  484,402 |               0 |           590 |
| + C1 MoE prototypes (K=4)                 | 0.820±0.009 | 0.418±0.086 |  487,606 |               0 |           595 |
| + C2 Unsup. TTA (on baseline)             | 0.812±0.034 | 0.379±0.140 |  484,482 |              80 |          1137 |
| ctrl: TTA, shuffled pseudo-labels         | 0.840±0.057 | 0.512±0.223 |  484,482 |              80 |          1138 |
| + C3 Edge purification                    | 0.824±0.047 | 0.498±0.164 |  485,107 |               0 |          1881 |
| ctrl: baseline + attack-aug (no purifier) | 0.799±0.084 | 0.401±0.229 |  484,402 |               0 |           604 |
| + C4 Adaptive PPR residual                | 0.834±0.020 | 0.492±0.107 |  484,534 |               0 |           627 |
| C1+C3+C4 combined                         | 0.825±0.034 | 0.408±0.148 |  488,443 |               0 |          1895 |
| Combined + C2 TTA                         | 0.819±0.047 | 0.379±0.177 |  488,523 |              80 |          1899 |

### amazon_upu  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.792±0.017 | 0.387±0.099 |  484,402 |               0 |           542 |
| + C1 MoE prototypes (K=4)                 | 0.793±0.011 | 0.373±0.084 |  487,606 |               0 |           547 |
| + C2 Unsup. TTA (on baseline)             | 0.796±0.016 | 0.352±0.107 |  484,482 |              80 |           615 |
| ctrl: TTA, shuffled pseudo-labels         | 0.795±0.013 | 0.363±0.100 |  484,482 |              80 |           615 |
| + C3 Edge purification                    | 0.758±0.016 | 0.376±0.093 |  485,107 |               0 |          2528 |
| ctrl: baseline + attack-aug (no purifier) | 0.771±0.041 | 0.343±0.163 |  484,402 |               0 |           556 |
| + C4 Adaptive PPR residual                | 0.795±0.015 | 0.399±0.089 |  484,534 |               0 |           577 |
| C1+C3+C4 combined                         | 0.753±0.018 | 0.341±0.105 |  488,443 |               0 |          2542 |
| Combined + C2 TTA                         | 0.750±0.017 | 0.295±0.119 |  488,523 |              80 |          2546 |

### reddit  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.573±0.020 | 0.037±0.001 |  484,402 |               0 |           548 |
| + C1 MoE prototypes (K=4)                 | 0.556±0.026 | 0.036±0.002 |  487,606 |               0 |           553 |
| + C2 Unsup. TTA (on baseline)             | 0.551±0.026 | 0.035±0.002 |  484,482 |              80 |           625 |
| ctrl: TTA, shuffled pseudo-labels         | 0.556±0.041 | 0.035±0.003 |  484,482 |              80 |           625 |
| + C3 Edge purification                    | 0.500±0.000 | 0.033±0.000 |  485,107 |               0 |          1453 |
| ctrl: baseline + attack-aug (no purifier) | 0.528±0.047 | 0.033±0.004 |  484,402 |               0 |           562 |
| + C4 Adaptive PPR residual                | 0.558±0.027 | 0.036±0.002 |  484,534 |               0 |           584 |
| C1+C3+C4 combined                         | 0.482±0.037 | 0.033±0.004 |  488,443 |               0 |          1467 |
| Combined + C2 TTA                         | 0.484±0.051 | 0.033±0.006 |  488,523 |              80 |          1471 |

### elliptic  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.394±0.007 | 0.076±0.001 |  484,402 |               0 |           821 |
| + C1 MoE prototypes (K=4)                 | 0.399±0.021 | 0.076±0.002 |  487,606 |               0 |           826 |
| + C2 Unsup. TTA (on baseline)             | 0.418±0.009 | 0.077±0.001 |  484,482 |              80 |          1062 |
| ctrl: TTA, shuffled pseudo-labels         | 0.417±0.009 | 0.077±0.001 |  484,482 |              80 |          1062 |
| + C3 Edge purification                    | 0.499±0.029 | 0.100±0.006 |  485,107 |               0 |           836 |
| ctrl: baseline + attack-aug (no purifier) | 0.372±0.011 | 0.074±0.002 |  484,402 |               0 |           835 |
| + C4 Adaptive PPR residual                | 0.395±0.011 | 0.075±0.002 |  484,534 |               0 |           912 |
| C1+C3+C4 combined                         | 0.499±0.087 | 0.099±0.019 |  488,443 |               0 |           921 |
| Combined + C2 TTA                         | 0.525±0.070 | 0.103±0.019 |  488,523 |              80 |          1138 |

### AVERAGE over targets  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.648±0.009 | 0.244±0.049 |  484,402 |               0 |           626 |
| + C1 MoE prototypes (K=4)                 | 0.642±0.009 | 0.226±0.040 |  487,606 |               0 |           630 |
| + C2 Unsup. TTA (on baseline)             | 0.644±0.010 | 0.211±0.060 |  484,482 |              80 |           860 |
| ctrl: TTA, shuffled pseudo-labels         | 0.652±0.021 | 0.247±0.075 |  484,482 |              80 |           860 |
| + C3 Edge purification                    | 0.645±0.016 | 0.252±0.063 |  485,107 |               0 |          1675 |
| ctrl: baseline + attack-aug (no purifier) | 0.617±0.037 | 0.213±0.094 |  484,402 |               0 |           640 |
| + C4 Adaptive PPR residual                | 0.646±0.012 | 0.251±0.042 |  484,534 |               0 |           675 |
| C1+C3+C4 combined                         | 0.640±0.025 | 0.220±0.058 |  488,443 |               0 |          1706 |
| Combined + C2 TTA                         | 0.644±0.027 | 0.202±0.070 |  488,523 |              80 |          1764 |