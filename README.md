# ResNet-AHA-Coal-Structure
Official PyTorch implementation of **ResNet-AHA** for coal body structure classification using electrical microresistivity imaging (ERMI) well logs.

## Overview
This network embeds SE module and Triplet Attention into ResNet-18 to alleviate feature competition during multi-scale fusion, aiming to identify four types of coal body structures from grayscale logging images.

### Coal structure categories
0_primary, 1_fractured, 2_granular, 3_mylonitic

## Environment
Install dependencies:
```bash
pip install -r requirements.txt