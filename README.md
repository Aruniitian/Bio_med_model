# Pneumonia Detection Using Chest X-ray Imaging

A deep learning-based system to automatically detect pneumonia from chest X-ray images using transfer learning with DenseNet121.

## Project Overview

This project implements a binary image classification model to classify chest X-ray images as either:
- **NORMAL**: Healthy lung condition
- **PNEUMONIA**: Pneumonia-infected lungs

## Dataset

- **Source**: Chest X-Ray Pneumonia Dataset
- **Structure**:
  - Training set: 5,216 images (NORMAL: 1,341 | PNEUMONIA: 3,875)
  - Validation set: 782 images (15% split from training)
  - Test set: 624 images (NORMAL: 234 | PNEUMONIA: 390)
- **Total**: ~5,800 labeled chest X-ray images organized in train/val/test splits

## Model Performance

### Test Set Metrics
- **Accuracy**: 91.35%
- **Precision**: 91.79%
- **Recall**: 94.62%
- **F1-Score**: 93.18%
- **ROC-AUC**: 0.9669

### Key Results
- Strong recall (94.62%) ensures high detection of pneumonia cases
- Solid precision (91.79%) minimizes false alarms
- Balanced F1-score (93.18%) indicates overall model reliability

## Model Architecture

**Transfer Learning with DenseNet121**
- Pre-trained on ImageNet weights
- Input size: 224×224×3
- Feature extraction via DenseNet121 base
- Global Average Pooling
- Dropout (0.35) for regularization
- Binary classification head with sigmoid activation

### Training Strategy
1. **Initial Training (10 epochs)**: Frozen base model, learning rate 1e-3
2. **Fine-tuning (6 epochs)**: Unfreeze top 80 layers, learning rate 1e-5
3. **Class Weighting**: Balanced weights to handle imbalanced training data
4. **Augmentation**: Rotation, zoom, shift, horizontal flip, brightness variation
5. **Callbacks**: ModelCheckpoint (best accuracy), EarlyStopping, ReduceLROnPlateau

## Outputs

Results are saved in the `result/` folder:
- `pneumonia_densenet121.keras` – Final trained model
- `best_pneumonia_densenet121.keras` – Best checkpoint
- `metrics.json` – Test set metrics
- `classification_report.csv` – Per-class performance details
- `training_curves.png` – Loss and accuracy evolution
- `confusion_matrix.png` – Prediction breakdown by class
- `roc_curve.png` – ROC curve with AUC score
- `class_distribution.png` – Dataset split visualization

## Usage

### Run the Full Pipeline
```bash
jupyter notebook pneumonia_detection.ipynb
```

### Load and Use the Trained Model
```python
import tensorflow as tf

model = tf.keras.models.load_model('result/pneumonia_densenet121.keras')

# Preprocess and predict on new images
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.densenet import preprocess_input

img = load_img('path/to/xray.jpg', target_size=(224, 224))
img_array = img_to_array(img)
img_array = preprocess_input(img_array)
img_array = tf.expand_dims(img_array, 0)

prediction = model.predict(img_array)
probability = prediction[0][0]
label = "PNEUMONIA" if probability > 0.5 else "NORMAL"
print(f"Prediction: {label} ({probability:.2%})")
```

## Requirements

```
tensorflow-cpu==2.17.1
scikit-learn
pandas
matplotlib
seaborn
pillow
jupyter
```

### Install Dependencies
```bash
pip install -r requirements.txt
```

## File Structure

```
Bio_med/
├── Dataset/
│   └── chest_xray/
│       ├── train/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       ├── val/
│       │   ├── NORMAL/
│       │   └── PNEUMONIA/
│       └── test/
│           ├── NORMAL/
│           └── PNEUMONIA/
├── result/
│   ├── pneumonia_densenet121.keras
│   ├── best_pneumonia_densenet121.keras
│   ├── metrics.json
│   ├── classification_report.csv
│   ├── training_curves.png
│   ├── confusion_matrix.png
│   ├── roc_curve.png
│   └── class_distribution.png
├── pneumonia_detection.ipynb
├── .gitignore
├── README.md
└── requirements.txt
```

## Key Features

1. **Transfer Learning**: Leverages pre-trained DenseNet121 for faster convergence and better performance
2. **Class Balancing**: Addresses dataset imbalance (PNEUMONIA: 3875 vs NORMAL: 1341)
3. **Data Augmentation**: Enhances generalization with rotation, zoom, shift, brightness variations
4. **Best Model Checkpointing**: Saves the model with highest validation accuracy
5. **Comprehensive Evaluation**: Accuracy, precision, recall, F1-score, ROC-AUC, and confusion matrix
6. **Visualization**: Training curves, confusion matrix, and ROC curve for interpretability

## Clinical Application

This model can assist radiologists and doctors in:
- Early detection of pneumonia from chest X-rays
- Improving diagnostic efficiency in healthcare settings
- Reducing false negatives through high recall (94.62%)
- Computer-aided diagnosis (CAD) systems

## Future Improvements

1. Increase validation set size for more stable model selection
2. Implement Grad-CAM for explainability and feature visualization
3. Optimize decision threshold based on clinical requirements (recall vs precision trade-off)
4. Ensemble methods combining multiple architectures (EfficientNet, ResNet, etc.)
5. Cross-validation for more robust performance estimation

## Author

**Aruniitian**  
GitHub: https://github.com/Aruniitian

## License

This project is provided for educational and research purposes.

---

**Last Updated**: April 15, 2026  
**Model Accuracy**: 91.35%
