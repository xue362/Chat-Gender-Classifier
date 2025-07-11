        
# Chinese Chat Gender Classifier Project

## Overview
This project implements a Chinese text-based gender classifier using transformer models. It can predict whether a given Chinese text was written by a male or female author.

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

## Release
- Pre-trained model weights are available in the [Releases](https://github.com/xue362/Chinese-Chat-Gender-Classifier/releases/tag/v0.1.with_weights) section

## Additional Notes
- This project was developed with assistance from AI tools


        
