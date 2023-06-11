# Knowledge Diffusion for Distillation (DiffKD)
Official implementation for paper "[Knowledge Diffusion for Distillation](https://arxiv.org/abs/2305.15712)" (DiffKD).

---

## Updates
### June 9, 2023
We released the core code of DiffKD. See `example.py` for usage.

---

## Reproducing our results

**We are refactoring our code and reproducing the results.**

### ImageNet

* Baseline settings (`R34-R101` and `R50-MBV1`):  
    ```
    CONFIG=configs/strategies/distill/TODO
    ```
    |Student|Teacher|DiffKD|MODEL|T_MODEL|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|72.20|`tv_resnet18`|`tv_resnet34`|[log](https://github.com/hunto/DiffKD/releases/download/v0.0.1/log_b1_diffkd_res34_res18.txt)|[ckpt](https://drive.google.com/file/d/19Wy5RCfpkAg9oUIiDUhfWtl8DjD712DN/view?usp=sharing)|
    |MobileNet V1 (70.13)|ResNet-50 (76.16)|73.24|`mobilenet_v1`|`tv_resnet50`|to be reproduced||

