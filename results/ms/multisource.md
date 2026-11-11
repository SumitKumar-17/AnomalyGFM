
### Target: amazon  (mean±std over 3 seeds; target domain never in the source pool)
| Method                                            | Stage         | AUROC       | AUPRC       |   Params |   GPU MB (incr.) |
|:--------------------------------------------------|:--------------|:------------|:------------|---------:|-----------------:|
| [1-src FB] Baseline (GCN+SVD, sup.)               | zero-shot     | 0.832±0.029 | 0.506±0.149 |  484,402 |              139 |
| [1-src FB] + C1 H2GCN                             | zero-shot     | 0.667±0.061 | 0.172±0.037 |  810,802 |              154 |
| [1-src FB] + C2 Hypernet align                    | zero-shot     | 0.360±0.031 | 0.051±0.004 |  507,514 |              134 |
| [1-src FB] + C3 SSL (NO labels)                   | zero-shot     | 0.591±0.102 | 0.093±0.024 |  484,002 |              133 |
| [1-src FB] C1+C2+C3 (sup+SSL)                     | zero-shot     | 0.466±0.070 | 0.061±0.011 |  833,914 |              154 |
| [4-src] Baseline (GCN+SVD, sup.)                  | zero-shot     | 0.611±0.089 | 0.131±0.090 |  484,402 |              935 |
| [4-src] + C1 H2GCN                                | zero-shot     | 0.548±0.116 | 0.111±0.069 |  810,802 |              940 |
| [4-src] + C2 Hypernet align                       | zero-shot     | 0.620±0.012 | 0.101±0.020 |  507,514 |              943 |
| [4-src] + C3 SSL (NO labels)                      | zero-shot     | 0.573±0.122 | 0.092±0.027 |  484,002 |             1778 |
| [4-src] C1+C2+C3 (sup+SSL)                        | zero-shot     | 0.530±0.012 | 0.071±0.002 |  833,914 |             1915 |
| [4-src] Base + prompt tuning (paper protocol)     | adapt (k=100) | 0.610±0.089 | 0.130±0.089 |       80 |              687 |
| [4-src] Base + C4 Graph-LoRA (paper protocol)     | adapt (k=100) | 0.610±0.090 | 0.131±0.090 |    4,832 |              705 |
| [4-src] Combined + C4 Graph-LoRA (paper protocol) | adapt (k=100) | 0.530±0.012 | 0.071±0.002 |   14,496 |              708 |
| [4-src] [sens.] prompt (30 ep, lr 0.001)          | adapt (k=100) | 0.576±0.077 | 0.106±0.056 |       80 |              687 |
| [4-src] [sens.] full GNN FT (30 ep, lr 0.001)     | adapt (k=100) | 0.449±0.033 | 0.057±0.003 |  164,002 |              706 |
| [4-src] [sens.] Graph-LoRA (30 ep, lr 0.001)      | adapt (k=100) | 0.448±0.102 | 0.060±0.015 |    4,832 |              705 |

### Target: amazon_upu  (mean±std over 3 seeds; target domain never in the source pool)
| Method                                            | Stage         | AUROC       | AUPRC       |   Params |   GPU MB (incr.) |
|:--------------------------------------------------|:--------------|:------------|:------------|---------:|-----------------:|
| [1-src FB] Baseline (GCN+SVD, sup.)               | zero-shot     | 0.785±0.020 | 0.390±0.141 |  484,402 |               86 |
| [1-src FB] + C1 H2GCN                             | zero-shot     | 0.713±0.062 | 0.253±0.111 |  810,802 |              105 |
| [1-src FB] + C2 Hypernet align                    | zero-shot     | 0.363±0.022 | 0.049±0.002 |  507,514 |               87 |
| [1-src FB] + C3 SSL (NO labels)                   | zero-shot     | 0.571±0.070 | 0.085±0.022 |  484,002 |               86 |
| [1-src FB] C1+C2+C3 (sup+SSL)                     | zero-shot     | 0.547±0.039 | 0.076±0.011 |  833,914 |              105 |
| [4-src] Baseline (GCN+SVD, sup.)                  | zero-shot     | 0.612±0.070 | 0.128±0.080 |  484,402 |              935 |
| [4-src] + C1 H2GCN                                | zero-shot     | 0.548±0.066 | 0.084±0.029 |  810,802 |              940 |
| [4-src] + C2 Hypernet align                       | zero-shot     | 0.617±0.018 | 0.121±0.002 |  507,514 |              944 |
| [4-src] + C3 SSL (NO labels)                      | zero-shot     | 0.576±0.086 | 0.079±0.018 |  484,002 |             1779 |
| [4-src] C1+C2+C3 (sup+SSL)                        | zero-shot     | 0.508±0.025 | 0.064±0.004 |  833,914 |             1916 |
| [4-src] Base + prompt tuning (paper protocol)     | adapt (k=100) | 0.612±0.070 | 0.128±0.080 |       80 |              161 |
| [4-src] Base + C4 Graph-LoRA (paper protocol)     | adapt (k=100) | 0.612±0.071 | 0.128±0.080 |    4,832 |              177 |
| [4-src] Combined + C4 Graph-LoRA (paper protocol) | adapt (k=100) | 0.507±0.025 | 0.064±0.003 |   14,496 |              179 |
| [4-src] [sens.] prompt (30 ep, lr 0.001)          | adapt (k=100) | 0.593±0.062 | 0.104±0.046 |       80 |              161 |
| [4-src] [sens.] full GNN FT (30 ep, lr 0.001)     | adapt (k=100) | 0.485±0.038 | 0.061±0.005 |  164,002 |              178 |
| [4-src] [sens.] Graph-LoRA (30 ep, lr 0.001)      | adapt (k=100) | 0.527±0.044 | 0.067±0.007 |    4,832 |              176 |

### Target: reddit  (mean±std over 3 seeds; target domain never in the source pool)
| Method                                            | Stage         | AUROC       | AUPRC       |   Params |   GPU MB (incr.) |
|:--------------------------------------------------|:--------------|:------------|:------------|---------:|-----------------:|
| [1-src FB] Baseline (GCN+SVD, sup.)               | zero-shot     | 0.572±0.032 | 0.037±0.003 |  484,402 |               89 |
| [1-src FB] + C1 H2GCN                             | zero-shot     | 0.569±0.046 | 0.045±0.016 |  810,802 |              108 |
| [1-src FB] + C2 Hypernet align                    | zero-shot     | 0.573±0.058 | 0.044±0.004 |  507,514 |               90 |
| [1-src FB] + C3 SSL (NO labels)                   | zero-shot     | 0.582±0.013 | 0.040±0.002 |  484,002 |               89 |
| [1-src FB] C1+C2+C3 (sup+SSL)                     | zero-shot     | 0.555±0.017 | 0.037±0.002 |  833,914 |              109 |
| [4-src] Baseline (GCN+SVD, sup.)                  | zero-shot     | 0.591±0.018 | 0.039±0.002 |  484,402 |              934 |
| [4-src] + C1 H2GCN                                | zero-shot     | 0.590±0.014 | 0.047±0.012 |  810,802 |              939 |
| [4-src] + C2 Hypernet align                       | zero-shot     | 0.467±0.004 | 0.029±0.000 |  507,514 |              943 |
| [4-src] + C3 SSL (NO labels)                      | zero-shot     | 0.610±0.035 | 0.046±0.005 |  484,002 |             2000 |
| [4-src] C1+C2+C3 (sup+SSL)                        | zero-shot     | 0.476±0.081 | 0.031±0.006 |  833,914 |             2136 |
| [4-src] Base + prompt tuning (paper protocol)     | adapt (k=100) | 0.590±0.018 | 0.039±0.002 |       80 |              161 |
| [4-src] Base + C4 Graph-LoRA (paper protocol)     | adapt (k=100) | 0.590±0.017 | 0.039±0.002 |    4,832 |              178 |
| [4-src] Combined + C4 Graph-LoRA (paper protocol) | adapt (k=100) | 0.476±0.080 | 0.031±0.006 |   14,496 |              181 |
| [4-src] [sens.] prompt (30 ep, lr 0.001)          | adapt (k=100) | 0.582±0.016 | 0.038±0.001 |       80 |              161 |
| [4-src] [sens.] full GNN FT (30 ep, lr 0.001)     | adapt (k=100) | 0.608±0.048 | 0.044±0.008 |  164,002 |              179 |
| [4-src] [sens.] Graph-LoRA (30 ep, lr 0.001)      | adapt (k=100) | 0.608±0.007 | 0.049±0.008 |    4,832 |              178 |