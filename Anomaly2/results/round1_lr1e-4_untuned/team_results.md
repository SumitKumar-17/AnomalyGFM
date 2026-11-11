
### amazon  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.834±0.022 | 0.475±0.121 |  484,402 |               0 |           590 |
| + C1 MoE prototypes (K=4)                 | 0.811±0.016 | 0.380±0.115 |  487,606 |               0 |           595 |
| + C2 Unsup. TTA (on baseline)             | 0.812±0.034 | 0.379±0.140 |  484,482 |              80 |          1137 |
| ctrl: TTA, shuffled pseudo-labels         | 0.842±0.060 | 0.521±0.229 |  484,482 |              80 |          1138 |
| + C3 Edge purification                    | 0.644±0.207 | 0.223±0.216 |  485,043 |               0 |          1848 |
| ctrl: baseline + attack-aug (no purifier) | 0.632±0.197 | 0.211±0.202 |  484,402 |               0 |           604 |
| + C4 Adaptive PPR residual                | 0.814±0.067 | 0.396±0.223 |  484,534 |               0 |           627 |
| C1+C3+C4 combined                         | 0.536±0.176 | 0.101±0.063 |  488,379 |               0 |          1862 |
| Combined + C2 TTA                         | 0.525±0.158 | 0.084±0.041 |  488,459 |              80 |          1866 |

### amazon_upu  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.792±0.017 | 0.387±0.099 |  484,402 |               0 |           542 |
| + C1 MoE prototypes (K=4)                 | 0.788±0.017 | 0.360±0.097 |  487,606 |               0 |           547 |
| + C2 Unsup. TTA (on baseline)             | 0.796±0.016 | 0.352±0.107 |  484,482 |              80 |           615 |
| ctrl: TTA, shuffled pseudo-labels         | 0.795±0.013 | 0.364±0.100 |  484,482 |              80 |           615 |
| + C3 Edge purification                    | 0.633±0.117 | 0.128±0.094 |  485,043 |               0 |           567 |
| ctrl: baseline + attack-aug (no purifier) | 0.635±0.117 | 0.120±0.076 |  484,402 |               0 |           556 |
| + C4 Adaptive PPR residual                | 0.787±0.039 | 0.341±0.164 |  484,534 |               0 |           577 |
| C1+C3+C4 combined                         | 0.547±0.117 | 0.089±0.058 |  488,379 |               0 |           597 |
| Combined + C2 TTA                         | 0.548±0.108 | 0.079±0.034 |  488,459 |              80 |           644 |

### reddit  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.573±0.020 | 0.037±0.001 |  484,402 |               0 |           548 |
| + C1 MoE prototypes (K=4)                 | 0.565±0.025 | 0.036±0.002 |  487,606 |               0 |           553 |
| + C2 Unsup. TTA (on baseline)             | 0.551±0.026 | 0.035±0.002 |  484,482 |              80 |           625 |
| ctrl: TTA, shuffled pseudo-labels         | 0.556±0.041 | 0.035±0.003 |  484,482 |              80 |           625 |
| + C3 Edge purification                    | 0.499±0.020 | 0.030±0.001 |  485,043 |               0 |           565 |
| ctrl: baseline + attack-aug (no purifier) | 0.498±0.022 | 0.030±0.001 |  484,402 |               0 |           562 |
| + C4 Adaptive PPR residual                | 0.535±0.020 | 0.034±0.001 |  484,534 |               0 |           584 |
| C1+C3+C4 combined                         | 0.445±0.023 | 0.027±0.001 |  488,379 |               0 |           595 |
| Combined + C2 TTA                         | 0.472±0.019 | 0.029±0.001 |  488,459 |              80 |           649 |

### elliptic  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.394±0.007 | 0.076±0.001 |  484,402 |               0 |           821 |
| + C1 MoE prototypes (K=4)                 | 0.394±0.018 | 0.076±0.002 |  487,606 |               0 |           826 |
| + C2 Unsup. TTA (on baseline)             | 0.418±0.009 | 0.077±0.001 |  484,482 |              80 |          1062 |
| ctrl: TTA, shuffled pseudo-labels         | 0.417±0.009 | 0.077±0.001 |  484,482 |              80 |          1062 |
| + C3 Edge purification                    | 0.462±0.019 | 0.091±0.004 |  485,043 |               0 |           836 |
| ctrl: baseline + attack-aug (no purifier) | 0.393±0.029 | 0.076±0.003 |  484,402 |               0 |           835 |
| + C4 Adaptive PPR residual                | 0.370±0.006 | 0.073±0.001 |  484,534 |               0 |           912 |
| C1+C3+C4 combined                         | 0.448±0.055 | 0.083±0.008 |  488,379 |               0 |           921 |
| Combined + C2 TTA                         | 0.444±0.055 | 0.083±0.007 |  488,459 |              80 |          1138 |

### AVERAGE over targets  (train: Facebook; zero-shot; mean±std over 5 seeds; all target nodes evaluated)
| method                                    | AUROC       | AUPRC       |   Params |   Updated @test |   Peak GPU MB |
|:------------------------------------------|:------------|:------------|---------:|----------------:|--------------:|
| Baseline AnomalyGFM                       | 0.648±0.009 | 0.244±0.049 |  484,402 |               0 |           626 |
| + C1 MoE prototypes (K=4)                 | 0.639±0.007 | 0.213±0.052 |  487,606 |               0 |           630 |
| + C2 Unsup. TTA (on baseline)             | 0.644±0.010 | 0.211±0.060 |  484,482 |              80 |           860 |
| ctrl: TTA, shuffled pseudo-labels         | 0.653±0.021 | 0.249±0.076 |  484,482 |              80 |           860 |
| + C3 Edge purification                    | 0.559±0.083 | 0.118±0.078 |  485,043 |               0 |           954 |
| ctrl: baseline + attack-aug (no purifier) | 0.540±0.075 | 0.109±0.069 |  484,402 |               0 |           640 |
| + C4 Adaptive PPR residual                | 0.626±0.027 | 0.211±0.094 |  484,534 |               0 |           675 |
| C1+C3+C4 combined                         | 0.494±0.060 | 0.075±0.026 |  488,379 |               0 |           994 |
| Combined + C2 TTA                         | 0.497±0.054 | 0.069±0.016 |  488,459 |              80 |          1074 |