# Official PyTorch implementation of **ResNet-AHA**
Intelligent Recognition of Coal Body Structure from Electrical Imaging Logs via an Alternating Hybrid Attention Mechanism

## Overview
This network embeds an SE module and Triplet Attention into ResNet-18 to alleviate inter-scale feature competition during multi-scale feature fusion. The model is designed to identify four types of coal body structures from grayscale electrical microresistivity imaging (ERMI) logging images.

### Coal structure categories
`0_primary`, `1_cataclastic`, `2_granulated`, `3_mylonitic`

## Environment
Install dependencies:
```bash
pip install -r requirements.txt
```
## Dataset Statement
Due to geological data confidentiality restrictions, raw field ERMI logging data cannot be publicly released.
The folder `demo_synthetic_dataset` contains synthetic demo images for testing and reproducing the model pipeline.
The synthetic data cannot represent the characteristics of actual coal body structures and is only used for code verification.

The directory structure strictly matches the reading path in training code:
```
demo_synthetic_dataset/
├─ train
│  ├─ 0_primary
│  ├─ 1_cataclastic
│  ├─ 2_granulated
│  └─ 3_mylonitic
├─ val
│  ├─ 0_primary
│  ├─ 1_cataclastic
│  ├─ 2_granulated
│  └─ 3_mylonitic
└─ test
   ├─ 0_primary
   ├─ 1_cataclastic
   ├─ 2_granulated
   └─ 3_mylonitic
```
Run `python make_demo_dataset.py` to generate synthetic images before model training.

## Input and Output

### Input
Preprocessed single-channel grayscale ERMI logging images. Images are arranged in category-specific folders for training, validation and testing.

### Output
1. Optimized model checkpoint `ResNet_AHA_best.pth` saved in the `models/` folder.
2. Quantitative metrics (Accuracy, Precision, Recall, F1-Score) and confusion matrix printed in the console.
3. Training loss and accuracy curve image saved as `training_curves.png`.
## Quick Start
1. Generate synthetic demo dataset
```bash
python make_demo_dataset.py
```
2. Train ResNet-AHA classification model
```bash
python train_resnet_aha.py
```
## Citation
If you find this repository useful, please cite our work once published:
```
[Will be updated after paper acceptance]
```
## License
This project is distributed under the MIT License. See the LICENSE file in this repository for full license information.
