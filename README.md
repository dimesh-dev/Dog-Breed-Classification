# Dog Breed Image Classification Model

This project demonstrates building a deep learning-based image classification model using TensorFlow and Keras to classify images of dog breeds. The model is trained on a dataset of 90+ classes of dog breeds, achieves strong accuracy, and is deployed as a simple web application using Streamlit.

<img width="1920" height="2040" alt="screencapture-localhost-8501-2025-09-17-16_07_07" src="https://github.com/user-attachments/assets/5900d7b0-c03e-4541-84d1-fb53bac36132" />


Demo: [Go to Streamlit webapp](https://dog-breed-classifier.streamlit.app/)

Model: [Go to HuggingFace](https://huggingface.co/dimeshanthoney/dog-breed-classifier)

## Credit to Original Tutorial

This project is initially based on TensorFlow image classification tutorials, which provided the foundational steps for preprocessing, model building, training, and initial Streamlit deployment. Subsequent enhancements were made to handle multi-class dog breed classification, integrate Hugging Face model hosting, and improve user experience.

## Project Overview

- **Objective**: Classify images into 90+ categories of dog breeds (e.g., Labrador Retriever, Beagle, Golden Retriever, Rottweiler).
- **Approach**:
  - Preprocess image data into arrays using TensorFlow.
  - Build a sequential Convolutional Neural Network (CNN) model with multiple layers.
  - Train the model using training and validation datasets with data augmentation.
  - Evaluate the model on test data and unseen images.
  - Deploy the model as a web app for interactive predictions, hosted online for easy access.
- **Key Features**:
  - Handles image resizing, normalization, and batching.
  - Uses validation data and augmentation to improve generalization.
  - Supports Top-5 predictions for better interpretability.
  - Web app supports drag-and-drop uploads or browsing local files.
  - Model hosted on Hugging Face for seamless integration.
  - Achieves strong classification accuracy across breeds.

## Dataset

The dataset consists of images of dog breeds, split into:
- **Data**: ~8,000+ images

Each breed has its own subfolder (e.g., "beagle", "pug", "labrador_retriever").

- **Download Link**: [Dogs Dataset]([http://vision.stanford.edu/aditya86/ImageNetDogs/](https://www.kaggle.com/datasets/kabilan03/dogbreedclassification))
- **Classes**: ['afghan_hound', 'beagle', 'labrador_retriever', 'golden_retriever', 'pug', 'rottweiler', 'german_shepherd', ... up to 90+ breeds]

Place the dataset in a folder named `dog_breeds` with subfolders for `train`, `test`, and `validation`.

## Requirements

- Python 3.8+
- Libraries:
  - TensorFlow
  - Keras (included in TensorFlow)
  - NumPy
  - Pandas
  - Matplotlib
  - Pillow
  - Streamlit
  - Hugging Face Hub (for model hosting)

Install dependencies:
```bash
pip install tensorflow streamlit numpy pandas matplotlib pillow huggingface_hub
```

## 📁 Project Structure
```bash
Dog-Breed-Classification/
│
├── app.py                # Streamlit web app
├── Image_classify.keras  # Saved trained model file
├── requirements.txt      # Dependencies
├── model-to-hf.py        # Upload model to Hugging Face
├── .env                  # Hugging Face token (ignored in GitHub)
├── notebooks/            # (Optional) Jupyter notebooks for training/evaluation
└── dataset/              # Dog images (not included in repo)
```
## 🔄 Workflow

### 1. Data Preprocessing
- Load images and resize to **224x224**  
- Normalize pixel values (**divide by 255**)  
- Apply augmentation (**flips, rotations, zooms**)  
- Shuffle and batch images for training  

---

### 2. Model Building & Training
- **CNN architecture**:
  - Rescaling layer  
  - Multiple Conv2D + MaxPooling layers  
  - Flatten + Dense layers  
  - Softmax output layer for 90+ classes  
- **Optimizer**: Adam  
- **Loss**: Categorical Crossentropy  
- **Metrics**: Accuracy & Top-5 Accuracy  
- Trained for multiple epochs until convergence  

---

### 3. Model Evaluation
- Evaluate on validation/test set  
- Track:
  - Confusion matrix  
  - Accuracy & loss curves  
  - Top-5 accuracy  

---

### 4. Deployment as Web App
- Streamlit app allows uploading an image  
- Model predicts **Top-5 breeds with probabilities**  
- Integrated Hugging Face Hub model for cloud hosting  

## ▶️ Run Locally (optional)

Clone the repo:
```bash
git clone https://github.com/your-username/dog-breed-classification.git
cd dog-breed-classification
```
Run Streamlit:
```bash
streamlit run app.py
```
