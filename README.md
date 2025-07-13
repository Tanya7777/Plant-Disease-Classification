# Plant-Disease-Classification


This is a deep learning–based web application that identifies plant leaf diseases from uploaded images. It utilizes a custom-trained **ResNet** model built using **PyTorch**, served via a **Flask backend**, and features a simple, responsive web interface for ease of use.

---

## Table of Contents

- [Overview](#overview)
- [Data Science Pipeline](#data-science-pipeline)
- [Features](#features)
- [Tech Stack](#tech-stack)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Usage](#usage)
- [Model Details](#model-details)
- [Future Enhancements](#future-enhancements)

---

## Overview

This project addresses the need for accessible, automated plant disease diagnosis using computer vision. It allows users to upload an image of a plant leaf and receive an accurate disease classification along with a confidence score.

The solution is tailored for agriculture professionals, students, and researchers looking for intelligent plant disease monitoring tools.

---

## Data Science Pipeline

This project applies a complete data science lifecycle, as outlined below:

### 1. Data Collection & Preparation
- Used publicly available datasets such as [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease).
- Performed class balancing, resizing, and normalization of images.
- Organized data into training, validation, and testing subsets.

### 2. Exploratory Data Analysis (EDA)
- Analyzed disease class distribution and sample imbalance.
- Visualized example images to understand class variability.

### 3. Modeling
- Built a custom **ResNet Convolutional Neural Network** using **PyTorch**.
- Fine-tuned model parameters and used data augmentation to improve generalization.

### 4. Model Evaluation
- Validated performance using accuracy and loss plots.
- Used confusion matrices to examine misclassification patterns (optional).

### 5. Deployment
- Wrapped model inference into a Flask API.
- Integrated with a front-end interface for user-friendly interaction.

This full pipeline makes the project not just a machine learning demo, but a fully realized **applied data science system**.

---

## Features

- Upload plant leaf images via a web interface
- Predict disease class using a ResNet model
- Responsive and modern UI with preview and transition effects
- Trained on augmented and labeled real-world datasets
- Flask-based API integration for real-time inference

---

## Tech Stack

| Area         | Technologies Used                           |
|--------------|----------------------------------------------|
| Frontend     | HTML, CSS, JavaScript                        |
| Backend      | Python, Flask                                |
| Deep Learning| PyTorch, Custom ResNet Architecture          |
| UI/UX        | Responsive Design, Basic Animations          |
| Deployment   | Localhost (optional: Render, Hugging Face Spaces) |

---

## Project Structure

```
plant-disease-prediction/
│
├── static/                # CSS, JS, and image assets
├── templates/             # HTML templates (upload, result)
├── model/                 # Trained PyTorch model (resnet.pth)
├── app.py                 # Main Flask application
├── utils.py               # Preprocessing and prediction functions
├── requirements.txt       # Python package dependencies
└── README.md              # Project documentation
```

---

## Installation

1. **Clone the repository**
```bash
git clone https://github.com/Tanya7777/plant-disease-prediction.git
cd plant-disease-prediction
```

2. **Install dependencies**
```bash
pip install -r requirements.txt
```

3. **Add the trained model**  
Place the `resnet.pth` file in the `model/` directory.

4. **Run the application**
```bash
python app.py
```

Then open your browser and go to http://localhost:5000

---

## Usage

- Visit the home page
- Upload a clear image of a plant leaf
- Click on “Predict”
- View predicted disease name and confidence score

---

## Model Details

- Model Type: ResNet (Custom, Pretrained Backbone)
- Input: 224x224 RGB images
- Output: Disease class probability scores
- Dataset: Publicly available [PlantVillage](https://www.kaggle.com/datasets/emmarex/plantdisease) dataset (with augmentation)
- Framework: PyTorch

---

## Future Enhancements

- Add model explainability via Grad-CAM visualizations
- Improve frontend design with dashboard analytics
- Add multilingual support for farmers
- Deploy to a cloud platform with GPU support
- Enable batch image upload and prediction

---


