Certainly! Here's an updated version of the README file tailored to your repository name and Python file name:

---

# Hairstyle Recommendation Through Image

This project aims to detect and classify face shapes using facial landmarks detected by Dlib's shape predictor. Based on the classified face shape, it recommends suitable hairstyles using pre-defined hairstyle images.

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Customization](#customization)
5. [Contributing](#contributing)


## Introduction

This Python-based project utilizes computer vision techniques to detect faces, classify their shapes, and recommend suitable hairstyles. It involves the following key components:

- **Face Detection**: Utilizes Haar Cascade Classifier to detect faces in images or live video feed.
- **Facial Landmark Detection**: Employs Dlib's shape predictor to identify facial landmarks.
- **Face Shape Classification**: Classifies face shapes based on the positions of facial landmarks.
- **Hairstyle Recommendation**: Recommends hairstyles corresponding to each classified face shape.

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/Hairstyle_recommend_through_image.git
   ```
   
2. Install the required dependencies:
   ```bash
   pip install numpy opencv-python dlib
   ```

## Usage

1. Run the script `face_hair1.py` using Python:
   ```bash
   python face_hair1.py
   ```

2. Choose an option:
   - **Option 1**: Live Photo Capture
     - Press the spacebar to capture a photo from the live video feed.
   - **Option 2**: Image Upload
     - Enter the path of the image containing a face.

3. Once the image or live capture is processed, the script will display the detected face shapes along with recommended hairstyles.

## Customization

To customize the recommended hairstyles or add more face shapes:
1. Replace the existing hairstyle images located in the `hairstyle_images` dictionary with your own images.
2. Modify the `classify_face_shape()` function to adjust classification thresholds or add new face shapes.

## Contributing

Contributions are welcome! If you have any suggestions, feature requests, or bug reports, please open an issue or submit a pull request.

