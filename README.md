# Waste Classification Using Deep Learning

## Project Overview

This project uses Deep Learning and a Convolutional Neural Network (CNN) to classify waste images into six categories:

- Cardboard
- Glass
- Metal
- Paper
- Plastic
- Trash

A Streamlit web application is also included, where users can upload an image and get a predicted waste category.

## Technologies Used

- Python
- TensorFlow
- Keras
- NumPy
- Matplotlib
- Pillow
- Streamlit

## Dataset

The dataset contains 2,527 images divided into six waste categories.

The images were divided into:

- 2,024 training images
- 503 validation images

## Model

A Convolutional Neural Network (CNN) was developed for image classification.

The input images are resized to 128 × 128 pixels.

## Results

The model was tested on waste images and was able to classify several images correctly. The validation accuracy varied across training runs.

## Streamlit Application

The trained model is connected to a Streamlit application.

Users can upload an image and the application displays the predicted waste category and confidence score.

## How to Run

Install the required libraries:

```bash
pip install tensorflow numpy matplotlib pillow streamlit