#🚗 Car Damage Detection Using Deep Learning

This project is a web-based application that classifies the type of damage on a car from an uploaded image. The system predicts whether the damage is a scratch, dent, or weather-related damage using a trained Convolutional Neural Network (CNN) model and displays the prediction with a confidence score.

The model is trained using Google Teachable Machine and deployed using Streamlit for real-time inference through a simple web interface.

🔍 Problem Statement

Manual inspection of vehicle damage is time-consuming and subjective, especially in insurance claim processing and vehicle assessments. This project aims to automate the initial damage classification process using computer vision and deep learning techniques.

✅ Features

Upload car images through a web interface

Automatic classification of damage type:

Scratch

Dent

Weather Damage

Displays prediction confidence score

Fast and lightweight deployment using Streamlit

🧠 Model Training

The dataset was created and labeled using Google Teachable Machine.

A CNN-based image classification model was trained using transfer learning.

The trained model was exported in Keras (.h5) format for local deployment.

Labels were exported using labels.txt.

⚙️ Tech Stack

Python

TensorFlow / Keras

Streamlit (Web App Framework)

NumPy

Pillow (PIL)

Google Teachable Machine (Model Training)

🖥️ Application Workflow

User uploads an image through the Streamlit interface.

Image is resized to 224×224, center-cropped, and normalized.

Preprocessed image is passed to the CNN model.

Model predicts damage category and confidence score.

Result is displayed on the web interface.
