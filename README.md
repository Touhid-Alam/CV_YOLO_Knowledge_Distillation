# README: Distilled-YOLOv11n-Leaf (DYL-Leaf) for Plant Leaf Disease Detection

## Overview
This repository contains the implementation of **DYL-Leaf (Distilled-YOLOv11n-Leaf)**, a lightweight and efficient model for detecting plant leaf diseases. The model is based on **YOLOv11n** and leverages **Knowledge Distillation (KD)** to achieve high performance with significantly fewer parameters compared to traditional models. The model is designed to detect 13 classes of potato, rice, and tomato leaf diseases from the **PlantVillage dataset**.

The distilled student model, **DYL-Leaf**, achieves **93.8% validation accuracy**, outperforming the teacher model (**YOLOv11n**) while using only **545,005 parameters** (compared to the teacher's 2.6M parameters). This makes it suitable for deployment in resource-constrained environments.

---

## Key Features
- **Lightweight Model**: The distilled student model (DYL-Leaf) has only **545,005 parameters**, making it highly efficient for deployment on edge devices.
- **High Performance**: Achieves **93.8% validation accuracy**, **94.00% precision**, **93.23% recall**, and **93.38% F1-score**.
- **Knowledge Distillation**: Utilizes KD to transfer knowledge from a larger teacher model (YOLOv11n) to a smaller student model (DYL-Leaf).
- **Saliency Maps**: Includes model interpretation using saliency maps to visualize the regions influencing the model's predictions.
- **Dataset**: Trained on a subset of the **PlantVillage dataset** containing **4,144 images** across **13 classes** of potato, rice, and tomato leaf diseases.

---

## Dataset
The dataset used in this study is a subset of the **PlantVillage dataset**, containing **4,144 images** across **13 classes** of potato, rice, and tomato leaf diseases. The dataset is available on [Kaggle](https://www.kaggle.com/datasets/touhidalam08/plantvillage-13classes).

### Dataset Classes
| Class Name                                      | Instances |
|-------------------------------------------------|-----------|
| Potato_Early_blight                             | 223       |
| Potato_Late_blight                              | 353       |
| Potato_healthy                                  | 152       |
| Rice_Bacterialblight                            | 339       |
| Rice_Blast                                      | 416       |
| Rice_Blight                                     | 213       |
| Rice_Brownspot                                  | 317       |
| Rice_Tungro                                     | 260       |
| Tomato_Late_blight                              | 407       |
| Tomato_Spider_mites_Two_spotted_spider_mite     | 324       |
| Tomato_Tomato_YellowLeaf_Curl_Virus             | 568       |
| Tomato_Tomato_mosaic_virus                      | 373       |
| Tomato_healthy                                  | 199       |

---

## Methodology
### 1. **Data Preprocessing**
The dataset was preprocessed using the following techniques:
- Resizing images to **256x256**.
- Random horizontal flipping.
- Random rotation (**10 degrees**).
- Brightness and contrast adjustment (**0.2**).
- Random grayscale conversion.
- Normalization of pixel values to the range **[0, 1]**.

### 2. **Model Architecture**
The **YOLOv11n** architecture consists of three main modules:
1. **Backbone**: Responsible for feature extraction using convolutional layers and the **C3k2** block.
2. **Neck**: Enhances feature representations using **C2PSA** (Cross Stage Partial with Spatial Attention) and **SPPF** (Spatial Pyramid Pooling-Fast) blocks.
3. **Head**: Performs final object classification and localization at multiple scales.

### 3. **Knowledge Distillation (KD)**
- The teacher model (**YOLOv11n**) is trained first.
- The student model (**DYL-Leaf**) is trained using KD to transfer knowledge from the teacher.
- The distillation loss combines **hard loss** (cross-entropy) and **soft loss** (Kullback-Leibler divergence with temperature scaling).

### 4. **Training**
- **Optimizer**: Adam.
- **Batch Size**: 32.
- **Learning Rate**: 0.001.
- **Epochs**: 100 (with early stopping after 30 epochs of no improvement).
- **Loss Function**: Combined hard and soft losses for KD.

---

## Results
### Performance Comparison
| Model               | Parameters | Validation Accuracy | Precision | Recall | F1-Score |
|---------------------|------------|---------------------|-----------|--------|----------|
| Teacher (YOLOv11n)  | 2.6M       | 92.9%              | 91.42%    | 90.00% | 90.42%   |
| Student (DYL-Leaf)  | 545,005    | 93.8%              | 94.00%    | 93.23% | 93.38%   |

### Class-wise Performance
The student model achieved high precision and recall for most classes, with particularly strong performance on tomato leaf diseases. However, **Rice Blight** remained a challenging class, indicating potential areas for future improvement.

---

## Model Interpretation
Saliency maps were used to interpret the model's decision-making process. These maps highlight the regions of the image that most influenced the model's predictions, providing insights into the model's focus areas.

---

## Usage
### Installation
1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/DYL-Leaf.git
   cd DYL-Leaf
