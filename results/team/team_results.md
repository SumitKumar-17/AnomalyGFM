
### Target: amazon  (train: Facebook; mean±std over 5 seeds; eval excludes the 100 tuned normal nodes)
| Method                                       | Stage         | AUROC       | AUPRC       |   AUROC(fusion) | FB-heldout AUROC   |   Trainable params |   Peak GPU MB |
|:---------------------------------------------|:--------------|:------------|:------------|----------------:|:-------------------|-------------------:|--------------:|
| Baseline AnomalyGFM (GCN+SVD, sup.)          | zero-shot     | 0.834±0.022 | 0.476±0.121 |           0.831 | 0.828              |            484,402 |           508 |
| + C1 H2GCN backbone                          | zero-shot     | 0.668±0.060 | 0.174±0.059 |           0.666 | 0.789              |            810,802 |           532 |
| + C2 Hypernet alignment                      | zero-shot     | 0.418±0.092 | 0.058±0.013 |           0.418 | 0.750              |            507,514 |           512 |
| + C3 SSL prototypes (NO labels)              | zero-shot     | 0.620±0.082 | 0.100±0.020 |           0.547 | 0.701              |            484,002 |           511 |
| C1+C2+C3 combined (sup+SSL)                  | zero-shot     | 0.529±0.105 | 0.083±0.040 |           0.533 | 0.675              |            833,914 |           533 |
| Base + prompt tuning (paper protocol)        | adapt (k=100) | 0.834±0.022 | 0.479±0.120 |           0.831 | -                  |                 80 |          1063 |
| Base + full GNN fine-tune (paper protocol)   | adapt (k=100) | 0.839±0.021 | 0.498±0.121 |           0.836 | -                  |            164,002 |          1083 |
| Base + C4 Graph-LoRA (paper protocol)        | adapt (k=100) | 0.834±0.022 | 0.477±0.123 |           0.831 | -                  |              4,832 |          1082 |
| Combined + C4 Graph-LoRA (paper protocol)    | adapt (k=100) | 0.529±0.105 | 0.083±0.040 |           0.533 | -                  |             14,496 |          1085 |
| [sens.] prompt tuning (30 ep, lr 0.001)      | adapt (k=100) | 0.851±0.049 | 0.569±0.165 |           0.848 | -                  |                 80 |          1064 |
| [sens.] full GNN fine-tune (30 ep, lr 0.001) | adapt (k=100) | 0.637±0.116 | 0.199±0.117 |           0.638 | -                  |            164,002 |          1084 |
| [sens.] Graph-LoRA (30 ep, lr 0.001)         | adapt (k=100) | 0.651±0.070 | 0.185±0.093 |           0.651 | -                  |              4,832 |          1083 |

### Target: amazon_upu  (train: Facebook; mean±std over 5 seeds; eval excludes the 100 tuned normal nodes)
| Method                                       | Stage         | AUROC       | AUPRC       |   AUROC(fusion) | FB-heldout AUROC   |   Trainable params |   Peak GPU MB |
|:---------------------------------------------|:--------------|:------------|:------------|----------------:|:-------------------|-------------------:|--------------:|
| Baseline AnomalyGFM (GCN+SVD, sup.)          | zero-shot     | 0.792±0.017 | 0.388±0.100 |           0.792 | 0.828              |            484,402 |           139 |
| + C1 H2GCN backbone                          | zero-shot     | 0.701±0.051 | 0.230±0.086 |           0.698 | 0.789              |            810,802 |           162 |
| + C2 Hypernet alignment                      | zero-shot     | 0.398±0.055 | 0.053±0.005 |           0.395 | 0.752              |            507,514 |           144 |
| + C3 SSL prototypes (NO labels)              | zero-shot     | 0.588±0.056 | 0.087±0.017 |           0.564 | 0.701              |            484,002 |           143 |
| C1+C2+C3 combined (sup+SSL)                  | zero-shot     | 0.543±0.030 | 0.075±0.010 |           0.547 | 0.676              |            833,914 |           162 |
| Base + prompt tuning (paper protocol)        | adapt (k=100) | 0.792±0.017 | 0.390±0.099 |           0.792 | -                  |                 80 |           215 |
| Base + full GNN fine-tune (paper protocol)   | adapt (k=100) | 0.795±0.016 | 0.403±0.092 |           0.795 | -                  |            164,002 |           232 |
| Base + C4 Graph-LoRA (paper protocol)        | adapt (k=100) | 0.793±0.017 | 0.391±0.098 |           0.793 | -                  |              4,832 |           231 |
| Combined + C4 Graph-LoRA (paper protocol)    | adapt (k=100) | 0.543±0.031 | 0.075±0.010 |           0.548 | -                  |             14,496 |           234 |
| [sens.] prompt tuning (30 ep, lr 0.001)      | adapt (k=100) | 0.797±0.010 | 0.479±0.052 |           0.798 | -                  |                 80 |           215 |
| [sens.] full GNN fine-tune (30 ep, lr 0.001) | adapt (k=100) | 0.586±0.054 | 0.095±0.032 |           0.588 | -                  |            164,002 |           232 |
| [sens.] Graph-LoRA (30 ep, lr 0.001)         | adapt (k=100) | 0.693±0.044 | 0.285±0.057 |           0.694 | -                  |              4,832 |           231 |

### Target: reddit  (train: Facebook; mean±std over 5 seeds; eval excludes the 100 tuned normal nodes)
| Method                                       | Stage         | AUROC       | AUPRC       |   AUROC(fusion) | FB-heldout AUROC   |   Trainable params |   Peak GPU MB |
|:---------------------------------------------|:--------------|:------------|:------------|----------------:|:-------------------|-------------------:|--------------:|
| Baseline AnomalyGFM (GCN+SVD, sup.)          | zero-shot     | 0.564±0.026 | 0.037±0.002 |           0.563 | 0.828              |            484,402 |           137 |
| + C1 H2GCN backbone                          | zero-shot     | 0.589±0.044 | 0.050±0.014 |           0.583 | 0.789              |            810,802 |           160 |
| + C2 Hypernet alignment                      | zero-shot     | 0.584±0.043 | 0.043±0.003 |           0.583 | 0.753              |            507,514 |           142 |
| + C3 SSL prototypes (NO labels)              | zero-shot     | 0.585±0.010 | 0.041±0.001 |           0.59  | 0.702              |            484,002 |           141 |
| C1+C2+C3 combined (sup+SSL)                  | zero-shot     | 0.550±0.016 | 0.037±0.001 |           0.549 | 0.676              |            833,914 |           161 |
| Base + prompt tuning (paper protocol)        | adapt (k=100) | 0.564±0.026 | 0.037±0.002 |           0.563 | -                  |                 80 |           208 |
| Base + full GNN fine-tune (paper protocol)   | adapt (k=100) | 0.563±0.026 | 0.037±0.002 |           0.562 | -                  |            164,002 |           226 |
| Base + C4 Graph-LoRA (paper protocol)        | adapt (k=100) | 0.563±0.026 | 0.037±0.002 |           0.562 | -                  |              4,832 |           225 |
| Combined + C4 Graph-LoRA (paper protocol)    | adapt (k=100) | 0.549±0.016 | 0.036±0.001 |           0.548 | -                  |             14,496 |           228 |
| [sens.] prompt tuning (30 ep, lr 0.001)      | adapt (k=100) | 0.568±0.019 | 0.037±0.001 |           0.567 | -                  |                 80 |           208 |
| [sens.] full GNN fine-tune (30 ep, lr 0.001) | adapt (k=100) | 0.596±0.025 | 0.040±0.003 |           0.596 | -                  |            164,002 |           226 |
| [sens.] Graph-LoRA (30 ep, lr 0.001)         | adapt (k=100) | 0.608±0.017 | 0.044±0.002 |           0.605 | -                  |              4,832 |           225 |