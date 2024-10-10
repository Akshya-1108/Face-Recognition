# Face Recognition using OpenCV

This project implements face recognition using OpenCV's inbuilt face recognition methods. The system can identify and verify faces by comparing them to a dataset of known faces. It uses deep learning-based models provided by OpenCV for accurate face recognition.

## Features
- Face recognition from live video feeds or static images
- Supports recognizing multiple faces at once
- Uses OpenCV’s inbuilt deep learning-based face recognition method
- Easy integration with other applications

## Technologies Used
- Python
- OpenCV

## How it Works
1. Load and preprocess the images of known faces to create face encodings.
2. Capture frames from a webcam or load an image containing faces.
3. Detect faces in the frame and compare them with the known face encodings.
4. Display the results, showing recognized faces with their names or "Unknown" labels.
