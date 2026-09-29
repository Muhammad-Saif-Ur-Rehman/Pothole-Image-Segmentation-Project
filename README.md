# 🛣️ Pothole Image Segmentation Project

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-4.5%2B-green)
![License](https://img.shields.io/badge/License-MIT-brightgreen)

A machine learning project designed to automatically detect and segment potholes in both static images and video streams. The model is optimized to perform segmentation tasks on real-time data, making it suitable for road maintenance monitoring, autonomous vehicle navigation, and infrastructure assessment.

## 🌟 Key Features

* **Real-Time Segmentation:** Capable of processing video feeds to segment potholes frame-by-frame with low latency.
* **Image & Video Support:** Flexible pipeline that accepts individual images, batch image folders, or video files (`.mp4`, `.avi`).
* **Custom Architecture:** Utilizes deep learning segmentation techniques to accurately isolate pothole boundaries from diverse road surfaces and lighting conditions.

## 🛠️ Tech Stack

* **Language:** Python
* **Computer Vision:** OpenCV
* **Deep Learning Framework:** PyTorch / TensorFlow *(Adjust based on your final weights)*
* **Data Manipulation:** NumPy, Pandas, Matplotlib (for visualization)

## 📂 Repository Structure

```text
├── data/
|   ├── train   # train data
|   └── Valid   # validation data                    
├── notebooks/
|   └── pothhole-image-segmentation.ipynb             # Jupyter notebooks for EDA, training, and evaluation
├── Outputs     #Stores outputs from model          
├── best.pt
├── app.py      # Streamlit application
├── requirements.txt       # Python dependencies
└── README.md              # Project documentation
