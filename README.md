# Tumor-from-Ultrasound-Image
![Python](https://img.shields.io/badge/python-3.7+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-blue.svg)

# Ultrasound Tumor Segmentation with DCFN+PBAC, U-Net, and Custom Model Comparison

This project offers a **graphical user interface (GUI) application** for segmenting tumors in ultrasound images using various models, including the **Dilated Fully Convolutional Network with Phase-Based Active Contour (DFCN+PBAC)** and **U-Net**, as well as a basic CNN model. Users can also upload a custom model to compare performance models.

## Features

- **Automatic Tumor Segmentation** with DCFN+PBAC, U-Net, and a simple CNN.
- **Custom Model Support**: Upload and compare your model against preloaded models.
- **Interactive GUI**: Built with Tkinter to load images, display segmentation results, and facilitate model comparison.
- **Segmentation Comparison**: View results and placeholder accuracy scores for each model side-by-side.

---
## ⚠️ Warning

**This software is not intended for medical diagnosis, treatment, or patient advice.** The code is designed for research purposes only, specifically for students and researchers to develop and contribute to  tumor segmentation. 

**Important:** 
- The model is incomplete and has low accuracy for real-world medical diagnosis.
- Do not rely on this software for any clinical decisions or patient care.

It can explore ultrasound image processing in tumor research as a learning tool.

## AI Assistance

Several AI-based tools were utilized to complete and debug the code, including tools for automating model tuning, error detection, and optimizing image processing functions.

---
## Installation

- Install the required dependencies:
    ```bash
    pip install numpy tkinter PIL tensorflow keras cv2 os
    ```
- or a Newer version of Python written in CMD
   ```bash
   py -m pip install numpy tkinter PIL tensorflow keras cv2 os
   ```
   
**Download or Prepare Model Files**
   - Place `dcfn_pbac_model.keras` and `unet_model.keras` in the root directory.
   - If these pre-trained models are unavailable, the code defaults to the simple CNN model for segmentation.

## Usage

1. **Using the Interface**
   - **Load Ultrasound Image**: Upload an ultrasound image via "Load Ultrasound Image". The simple CNN model will automatically segment the image and display results.
   - **View DCFN+PBAC and U-Net Results**: The DCFN+PBAC and U-Net models (if available) apply their segmentation to the uploaded image for comparison.
   - **Upload Custom Model**: Use "Upload Model for Comparison" to load a custom model that will segment the image.
   - **Compare Results**: Segmentation results and accuracy estimates from each model are displayed side-by-side for easy comparison.

## Implemented Models

1. **Dilated Fully Convolutional Network with Phase-Based Active Contour (DFCN+PBAC)**
   - This model combines dilated convolutions and phase-based contouring for accurate tumor segmentation in noisy ultrasound images.
   - **Reference**: [Dilated Fully Convolutional Network for Ultrasound Tumor Segmentation](https://doi.org/10.1002/acm2.13863)
   
2. **U-Net**
   - U-Net is widely used in biomedical image segmentation, providing robust segmentation even with limited data.
   - **Reference**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1016/j.ultrasmedbio.2020.06.015)

3. **Simple CNN Model**
   - This lightweight CNN model provides fast, straightforward segmentation and requires no pre-trained weights.

4. **Custom Model Support**
   - Users can upload additional models, which the application will directly integrate for comparison with the preloaded models.

## References

This project implements techniques from several fundamental research studies:

1. **Dilated Fully Convolutional Network (DFCN) and Phase-Based Active Contour (PBAC)**
   - **Paper**: [Dilated Fully Convolutional Network for Ultrasound Tumor Segmentation](https://doi.org/10.1002/acm2.13863)
   - **Summary**: DFCN with PBAC enhances segmentation accuracy in ultrasound images by integrating dilated convolutions and phase-based contouring.

2. **U-Net for Biomedical Image Segmentation**
   - **Paper**: [U-Net: Convolutional Networks for Biomedical Image Segmentation](https://doi.org/10.1016/j.ultrasmedbio.2020.06.015)
   - **Summary**: A leading architecture for medical image segmentation, particularly effective on small datasets.

3. **Comparison of Deep Learning Models for Ultrasound Image Analysis**
   - **Paper**: [Deep Residual Networks for Tumor Segmentation in Ultrasound](https://doi.org/10.1016/j.patcog.2018.02.012)
   - **Summary**: This study benchmarks model performance in ultrasound image segmentation, providing reference points for models included in this project.

## License
- This project is licensed under the MIT License. See the LICENSE file for details.
- This project was inspired by tutorials on working with image processing in Python. Thanks to the Python community and scientific libraries like `numpy`, `tkinter`, `PIL`, `tensorflow`, `keras`, `cv2` and `os` for making such projects possible.
---
## Contributing

I welcome researchers, students, and developers to contribute to this project, which aims to enhance the framework for **Tumor-from-Ultrasound-Image** using machine learning. While the current code provides a foundation with DCFN+PBAC, U-Net, there is potential to improve model accuracy, usability, and overall functionality.

### How You Can Contribute:
- **Improve Model Accuracy**: Enhance the current segmentation models or explore alternative deep learning architectures.
- **Optimize Feature Extraction**: Implement advanced image preprocessing or segmentation techniques to improve model performance.
- **Expand Model Support**: Add compatibility for additional pre-trained models commonly used in ultrasound or medical imaging.
- **Refine GUI Usability**: Enhance the GUI for better accessibility, adding intuitive controls and improved visualization options.
- **Dataset Integration**: Integrate real ultrasound datasets for more robust training and validation of model performance.
- **Segmentation Metrics**: Add support for additional performance metrics to assess segmentation quality across models better.

### How to Get Started:
1. **Fork the Repository**: Create your copy of the project by clicking "Fork" at the top of this page.
2. **Make Your Improvements**: Work on your chosen area and commit changes.
3. **Submit a Pull Request**: Once complete, open a pull request with a detailed explanation of your contributions.

Your contributions are highly valued, and collaborative development is encouraged to make this project a powerful research tool. We can build a more accurate and accessible ultrasound tumor segmentation system.
