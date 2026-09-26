# Brain Tumor MRI Classification — Hybrid Multi-Model Ensemble

A deep learning project for classifying brain MRI scans into four categories: **Glioma**, **Meningioma**, **No Tumor**, and **Pituitary**.

Instead of relying on a single pretrained network, this project fuses **six different feature-extraction backbones** into one hybrid model. Each architecture brings different inductive biases (depth, residual connections, dense connectivity, compound scaling, etc.), allowing the model to capture complementary patterns in MRI textures.

## Models Used

| Branch                    | Type                        | Pretrained |
|---------------------------|-----------------------------|----------|
| Custom CNN                | From-scratch CNN            | No       |
| EfficientNetB3            | Compound-scaled EfficientNet| ImageNet |
| VGG16                     | Classic deep CNN            | ImageNet |
| ResNet50                  | Residual Network            | ImageNet |
| DenseNet121               | Densely Connected Network   | ImageNet |
| MedicalNet-inspired       | 2D Residual CNN             | No       |

> **Note:** The real MedicalNet (Med3D) is a 3D ResNet pretrained on CT volumes in PyTorch. Since this dataset consists of 2D MRI slices, a 2D residual network inspired by MedicalNet’s architecture is used instead.

## Key Features

- Hybrid multi-branch architecture with feature fusion
- Two-phase training (frozen backbones → fine-tuning)
- Weighted soft-voting ensemble for comparison
- Full evaluation: Classification Report, Confusion Matrix, ROC-AUC
- Grad-CAM explainability
- Stratified train/validation split + class weights
- Data augmentation pipeline

## Results

| Method                        | Test Accuracy |
|-------------------------------|---------------|
| **Hybrid Fusion Model**       | **92.81%**    |
| Weighted Soft-Voting Ensemble | 88.00%        |

The hybrid fusion approach significantly outperformed the soft-voting ensemble of individual models.

## Dataset

- **Source:** [Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Classes:** Glioma, Meningioma, No Tumor, Pituitary
- **Split:** Train / Validation / Test (stratified)

## Tech Stack

- TensorFlow / Keras
- scikit-learn
- NumPy, Matplotlib, Seaborn, Pillow

## Project Structure

1. Environment setup & imports  
2. Configuration  
3. Data loading & stratified split  
4. Exploratory data analysis  
5. `tf.data` input pipeline  
6. Branch builders (6 models)  
7. Hybrid fusion model  
8. Class weights & callbacks  
9. Two-phase training  
10. Training curves  
11. Evaluation on test set  
12. Individual backbone baselines  
13. Weighted soft-voting ensemble  
14. Grad-CAM  
15. Model saving & loading  
16. Single-image & batch inference  
17. Appendix: How to plug in true 3D MedicalNet  

## How to Run

1. Download the dataset and update the paths in the configuration section.
2. Install dependencies:
   ```bash
   pip install tensorflow scikit-learn matplotlib seaborn pillow
