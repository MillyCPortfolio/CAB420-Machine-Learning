# Assessment 1A - Achieved 79.5/90 (High Distinciton)

This assessment consists of three problems focusing on regression, classification, and deep neural networks. Below is a brief overview of each task.  

## **Problem 1: Regression**  
**Objective:** Predict violent crimes per capita using socio-economic data from the 1990 US Census.  
**Tasks:**  
1. Train and evaluate:  
   - Linear Regression  
   - Ridge Regression (with λ tuning on validation set)  
   - LASSO Regression (with λ tuning on validation set)  
2. Compare models based on:  
   - Predictive performance  
   - Model complexity  
   - Ethical considerations in socio-economic modeling  

---  

## **Problem 2: Classification**  
**Objective:** Classify land type from spectral reflectance data (4 classes: Sugi forest, Hinoki forest, Mixed deciduous forest, Other).  
**Models:**  
1. **K-Nearest Neighbors (KNN)**  
2. **Random Forest**  
3. **SVM Ensemble**  

---  

## **Problem 3: Deep Neural Networks**  
**Objective:** Classify digits in the SVHN dataset using limited training data (1,000 samples).  
**Tasks:**  
1. Train a DCNN **without data augmentation**.  
2. Train a DCNN **with data augmentation** (e.g., rotations, shifts).  
3. Compare both DCNNs and a baseline SVM on:  
   - Accuracy  
   - Training/inference time

---

# Assessment 1B - Achieved 77/90 (High Distinction)

This assessment consists of two problems focusing on person re-identification and multi-task learning with fine-tuning. Below is a structured overview of each task.

## **Problem 1: Person Re-Identification**  
**Objective:** Match probe images to gallery images using both traditional and deep learning approaches.  
**Dataset:** Market-1501 subset (5,933 training images, 301 test identities split into Gallery/Probe sets).  

### **Tasks:**  
1. **Non-Deep Learning Method**  
   - Apply dimension reduction (e.g., PCA, LDA).  
   - Evaluate using Top-1/5/10 accuracy and CMC curves.  

2. **Deep Learning Method**  
   - Implement metric learning (e.g., Siamese networks, triplet loss).  
   - Evaluate using Top-1/5/10 accuracy and CMC curves.  

3. **Comparison**  
   - Contrast performance, computational efficiency, and failure cases.  

---

## **Problem 2: Multi-Task Learning & Fine-Tuning**  
**Objective:** Simultaneously classify breeds and segment foreground/background in pet images.  
**Dataset:** Oxford-IIIT Pets (37 breeds, semantic masks).  

### **Tasks:**  
1. **From-Scratch Model**  
   - Design a custom CNN for classification + segmentation.  

2. **Fine-Tuned Model**  
   - Adapt MobileNetV3Small for both tasks.  

3. **Evaluation**  
   - Compare performance on classification (accuracy) and segmentation (IoU).  
   - Analyze failures and propose mitigations.

---

# Assessment 2 (AI-Generated Image Detection) - Acheived 98/100 (High Distinction)

## **Objective**  
To evaluate machine learning methods for distinguishing AI-generated images from human-created content, addressing ethical and security concerns in digital authenticity.

---

## **Tasks**  
1. **Model Development & Training**:  
   - Implemented four approaches:  
     - Traditional: **SVM with HOG+LDA**  
     - Deep Learning: **VGG-Style CNN**, **Siamese Triplet Network**, **Vision Transformer (ViT)**  
   - Trained on a balanced dataset (Shutterstock human vs. AI-generated images).  

2. **Evaluation**:  
   - Compared performance using:  
     - Accuracy, F1-score, error rate  
     - Training time and computational efficiency  
   - Analyzed failure cases and model interpretability (e.g., Grad-CAM for CNN).  

3. **Ethical Consideration**:  
   - Discussed implications of false positives/negatives in real-world applications (e.g., misinformation, fraud).  

---

## **Key Findings**  
1. **Performance Hierarchy**:  
   - **VGG-Style CNN** achieved **99.6% accuracy** (best overall), leveraging hierarchical feature learning for artifact detection.  
   - **Siamese Triplet Network** performed well (**97.82% accuracy**) but required additional k-NN classification.  
   - **ViT** showed promise (**91.76% accuracy**) but was computationally expensive.  
   - **SVM (HOG+LDA)** lagged (**78.86% accuracy**), highlighting limitations of handcrafted features. 
