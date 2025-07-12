        
# Chinese Chat Gender Classifier

## Overview
This project implements a Chinese text-based gender classifier using transformer models([Chinese Bert](https://huggingface.co/hfl/chinese-roberta-wwm-ext)). It can predict whether a given Chinese text series was written by a male or female author.

## Requirements
- Python 3.7+
- PyTorch
- Transformers library
- pandas
- numpy

## Usage
1. **Training**
```bash
python train.py
```

2. **Evaluation**
```bash
python test.py
```

3. **Prediction**
- Interactive mode:
```bash
python predict.py
```
- Batch mode (recommended):
```bash
python predict.py --file input.csv --column message --output predictions.csv
```
- It is also recommended to give **over 50** messages as input.

## Release
- A pre-trained model is available in the [Releases](https://github.com/xue362/Chinese-Chat-Gender-Classifier/releases/tag/v0.1.with_weights) section.

## Dataset
-  Training data were collected from part of my wechat logs, which is worth noting that it can be inaccurate when predicting.
-  The dataset will **NOT** be made public due to privacy issue.

## Additional Notes
- Part, if not all, of this project was developed with AI.
- Current issue: **Nijigen** is more likely to be classified as female.


        
