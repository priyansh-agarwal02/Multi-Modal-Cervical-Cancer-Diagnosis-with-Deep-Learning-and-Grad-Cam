# Multi-Modal Cervical Cancer Diagnosis with Deep Learning and Grad-CAM

This project implements a multi-modal approach for cervical cancer diagnosis using deep learning models and Grad-CAM visualization. The code includes data preprocessing, model training, evaluation, and ensemble techniques to improve classification accuracy.

---

## **Implementation Overview**

### **1. Data Preprocessing**
- **CLAHE & Median Filter** ([CLAHE & Median Filter.ipynb](CLAHE%20&%20Median%20Filter.ipynb)):
  - Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) and Median Filtering to enhance image quality.
  - Converts `.bmp` images to `.png` format and saves them in the `Median and clahe Filters` directory.

- **NLM Filter** ([NLM filter.ipynb](NLM%20filter.ipynb)):
  - Applies Non-Local Means (NLM) filtering for noise removal.
  - Saves filtered images in the `NLM Filters` directory.

### **2. Model Training**
This project uses transfer learning to fine-tune pre-trained models for cervical cancer classification. The following models were used:

#### **DenseNet169 Model** ([DenseNet169_Model.ipynb](Model/DenseNet169_Model.ipynb)):
- **Transfer Learning Process**:
  - Loaded the pre-trained DenseNet169 model with ImageNet weights.
  - Froze the initial layers to retain pre-trained features.
  - Added custom layers on top, including:
    - GlobalAveragePooling2D
    - Dense layers with BatchNormalization and Dropout for regularization.
    - Final Dense layer with softmax activation for classification.
  - Fine-tuned the model by unfreezing some layers and training with a reduced learning rate.
- **Training Details**:
  - Data augmentation was applied to improve generalization.
  - Trained for 25 epochs with early stopping and learning rate reduction.
  - Saved the trained model as `DenseNet169_model.h5`.

#### **ResNet50V2 Model** ([ResNet50v2_Model.ipynb](Model/ResNet50v2_Model.ipynb)):
- **Transfer Learning Process**:
  - Loaded the pre-trained ResNet50V2 model with ImageNet weights.
  - Froze the initial convolutional blocks.
  - Added custom layers for classification, similar to DenseNet169.
  - Fine-tuned the model by unfreezing the later layers.
- **Training Details**:
  - Used data augmentation and trained for 25 epochs.
  - Saved the trained model as `ResNet50V2_model.h5`.

#### **ResNet101V2 Model** ([ResNet101_Model.ipynb](Model/ResNet101_Model.ipynb)):
- **Transfer Learning Process**:
  - Loaded the pre-trained ResNet101V2 model with ImageNet weights.
  - Froze the initial layers and added custom layers for classification.
  - Fine-tuned the model by unfreezing specific layers.
- **Training Details**:
  - Trained with early stopping and learning rate reduction.
  - Saved the trained model as `ResNet101V2_model.h5`.

#### **DenseNet121 Model** ([DenseNet121_Model.ipynb](Model/DenseNet121_Model.ipynb)):
- **Transfer Learning Process**:
  - Similar to DenseNet169, but used DenseNet121 architecture.
  - Fine-tuned the model by unfreezing layers and training with a reduced learning rate.
- **Training Details**:
  - Saved the trained model as `DenseNet121_model.h5`.

#### **XceptionNet Model** ([XceptionNet_model.ipynb](Model/XceptionNet_model.ipynb)):
- **Transfer Learning Process**:
  - Loaded the pre-trained XceptionNet model with ImageNet weights.
  - Froze the initial layers and added custom layers for classification.
  - Fine-tuned the model by unfreezing specific layers.
- **Training Details**:
  - Saved the trained model as `XceptionNet_model.h5`.

#### **InceptionResNetV2 Model** ([InceptionResNetV2_model.ipynb](Model/InceptionResNetV2_model.ipynb)):
- **Transfer Learning Process**:
  - Loaded the pre-trained InceptionResNetV2 model with ImageNet weights.
  - Froze the initial layers and added custom layers for classification.
  - Fine-tuned the model by unfreezing specific layers.
- **Training Details**:
  - Saved the trained model as `InceptionResNetV2_model.h5`.

### **3. Ensemble Approach**
- **Ensemble of Models** ([Ensemble_approach.ipynb](Ensemble_approach.ipynb)):
  - Combines predictions from multiple pre-trained models (ResNet50V2, ResNet101, DenseNet121, DenseNet169, XceptionNet, InceptionResNetV2).
  - Uses majority voting to determine the final prediction.
  - Includes Grad-CAM visualization for interpretability.

### **4. Grad-CAM Visualization**
- **Grad-CAM Implementation** ([Ensemble_approach.ipynb](Ensemble_approach.ipynb)):
  - Generates heatmaps to visualize the regions of the image that contribute most to the model's predictions.

---

## **Software Used**
- **Programming Language**: Python
- **Libraries**:
  - TensorFlow/Keras: For deep learning model implementation.
  - OpenCV: For image processing.
  - NumPy & Pandas: For data manipulation.
  - Matplotlib & Seaborn: For visualization.
  - Scikit-learn: For evaluation metrics.

---

## **Instructions to Use the Code**

### **1. Setup Environment**
- Install the required libraries:
  ```bash
  pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn

---
.
├── CLAHE & Median Filter.ipynb
├── NLM filter.ipynb
├── [Ensemble_approach.ipynb](http://_vscodecontentref_/2)
├── Model/
│   ├── [DenseNet169_Model.ipynb](http://_vscodecontentref_/3)
│   ├── [ResNet50v2_Model.ipynb](http://_vscodecontentref_/4)
│   ├── [ResNet101_Model.ipynb](http://_vscodecontentref_/5)
│   ├── [DenseNet121_Model.ipynb](http://_vscodecontentref_/6)
│   ├── [XceptionNet_model.ipynb](http://_vscodecontentref_/7)
│   └── [InceptionResNetV2_model.ipynb](http://_vscodecontentref_/8)
├── Herlev Dataset/
│   ├── train/
│   ├── test/
├── Median and clahe Filters/
├── NLM Filters/
└── [README.md](http://_vscodecontentref_/9)

## **Instructions to Use the Code**

### **1. Setup Environment**
- Install the required libraries:
  ```bash
  pip install tensorflow opencv-python numpy pandas matplotlib seaborn scikit-learn



## Key Features
# Data Preprocessing: Enhances image quality using CLAHE, Median, and NLM filters.
# Deep Learning Models: Implements state-of-the-art architectures for classification.
# Transfer Learning: Finetuning of models
# Ensemble Learning: Combines multiple models for improved accuracy.
# Grad-CAM: Provides interpretability by highlighting important regions in the image.