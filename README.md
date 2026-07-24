# Official PyTorch implementation of **ResNet-AHA** for coal body structure classification using electrical microresistivity imaging (ERMI) well logs.

## Overview
This network embeds SE module and Triplet Attention into ResNet-18 to alleviate feature competition during multi-scale fusion, aiming to identify four types of coal body structures from grayscale logging images.

### Coal structure categories
0_primary, 1_fractured, 2_granular, 3_mylonitic

## Environment
Install dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Statement
Due to geological data confidentiality restrictions, raw field ERMI logging data cannot be publicly released.
The folder `demo_synthetic_dataset` contains synthetic demo images for testing and reproducing the model pipeline.
The directory structure strictly matches the reading path in training code:
```
demo_synthetic_dataset/
├─ train
│  ├─ 0_原生
│  ├─ 1_碎裂
│  ├─ 2_碎粒
│  └─ 3_糜棱
├─ val
│  ├─ 0_原生
│  ├─ 1_碎裂
│  ├─ 2_碎粒
│  └─ 3_糜棱
└─ test
   ├─ 0_原生
   ├─ 1_碎裂
   ├─ 2_碎粒
   └─ 3_糜棱
```
## Quick Start
1. Preprocess synthetic dataset
```bash
python make_demo_dataset.py
```
2. Train ResNet-AHA classification model
```bash
python train_resnet_aha.py
```
## License
This project is distributed under the MIT License. See the LICENSE file in this repository for full license information.
