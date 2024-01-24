# Knowledge Diffusion for Distillation (DiffKD)
Official implementation for paper "[Knowledge Diffusion for Distillation](https://arxiv.org/abs/2305.15712)" (DiffKD), NeurIPS 2023

---

## Reproducing our results

```shell
git clone https://github.com/hunto/DiffKD.git --recurse-submodules
cd DiffKD
```

**The implementation of DiffKD is in** [classification/lib/models/losses/diffkd](https://github.com/hunto/image_classification_sota/tree/main/lib/models/losses/diffkd).

* classification: prepare your environment and datasets following the `README.md` in `classification`.

### ImageNet

```
cd classification
sh tools/dist_train.sh 8 ${CONFIG} ${MODEL} --teacher-model ${T_MODEL} --experiment ${EXP_NAME}
```

Example script for reproducing DiffKD on ResNet-34 teacher and ResNet-18 student with B1 baseline setting:
```
sh tools/dist_train.sh 8 configs/strategies/distill/diffkd/diffkd_b1.yaml tv_resnet18 --teacher-model tv_resnet34 --experiment diffkd_res34_res18
```

* Baseline settings (`R34-R18` and `R50-MBV1`):  
    ```
    CONFIG=configs/strategies/distill/TODO
    ```
    |Student|Teacher|DiffKD|MODEL|T_MODEL|Log|Ckpt|
    |:--:|:--:|:--:|:--:|:--:|:--:|:--:|
    |ResNet-18 (69.76)|ResNet-34 (73.31)|72.20|`tv_resnet18`|`tv_resnet34`|[log](https://github.com/hunto/DiffKD/releases/download/v0.0.1/log_b1_diffkd_res34_res18.txt)|[ckpt](https://drive.google.com/file/d/19Wy5RCfpkAg9oUIiDUhfWtl8DjD712DN/view?usp=sharing)|
    |MobileNet V1 (70.13)|ResNet-50 (76.16)|73.24|`mobilenet_v1`|`tv_resnet50`|to be reproduced||


## License  
This project is released under the [Apache 2.0 license](LICENSE).

## Citation 
```
@article{huang2023knowledge,
  title={Knowledge Diffusion for Distillation},
  author={Huang, Tao and Zhang, Yuan and Zheng, Mingkai and You, Shan and Wang, Fei and Qian, Chen and Xu, Chang},
  journal={arXiv preprint arXiv:2305.15712},
  year={2023}
}
```