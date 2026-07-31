# Official PyTorch implementation of **ResNet-AHA** for coal body structure classification using electrical microresistivity imaging (ERMI) well logs.

## Overview
This network embeds SE module and Triplet Attention into ResNet-18 to alleviate feature competition during multi-scale fusion, aiming to identify four types of coal body structures from grayscale logging images.

### Coal structure categories
0_primary, 1_cataclastic, 2_granulated, 3_mylonitic

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

## Input and Output
### Input
Preprocessed single-channel grayscale ERMI logging images. Images are arranged in category-specific folders for training, validation and testing.

Output
Optimized model checkpoint ResNet_AHA_best.pth saved in the models/ folder.
Quantitative metrics (Accuracy, Precision, Recall, F1-Score) and confusion matrix printed in the console.
Training loss and accuracy curve image saved as training_curves.png.
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
