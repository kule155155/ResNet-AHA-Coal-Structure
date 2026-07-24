# Official PyTorch implementation of **ResNet-AHA** for coal body structure classification using electrical microresistivity imaging (ERMI) well logs.

## Overview
This network embeds SE module and Triplet Attention into ResNet-18 to alleviate feature competition during multi-scale fusion, aiming to identify four types of coal body structures from grayscale logging images.
This code repository supports the research published in Computers & Geosciences. All training scripts, dataset generation code and synthetic test dataset are provided for reproducibility.

### Coal structure categories
0_primary (0_原生), 1_fractured (1_碎裂), 2_granular (2_碎粒), 3_mylonitic (3_糜棱)

## Environment & Dependencies
All required packages are listed in `requirements.txt`.
Install dependencies via pip:
```bash
pip install -r requirements.txt
