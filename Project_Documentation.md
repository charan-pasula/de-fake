# PROJECT REPORT: De-Fake
## An Advanced Deepfake Detection System using Vision Transformers (ViT)

### Team Members:
1. charan pasula - 24P81A05B3
2. kamsani srujith kumar - 24P81A0596
3. rachakonda sanjana - 24P81A05B7
4.ponugoti bhavya - 24P81A05B4
5.telikapalli chandana - 24P81A05C3

Under the Guidance of:dr.G.keerthi madam

---

## 1. Abstract
The rapid advancement of generative AI has led to the proliferation of highly realistic, artificially generated media known as "deepfakes." These synthetic images pose a significant threat to digital trust, security, and misinformation. This project, **De-Fake**, introduces a highly accurate detection system leveraging the state-of-the-art **Vision Transformer (ViT)** architecture. Unlike traditional Convolutional Neural Networks (CNNs), the ViT model processes images as sequences of localized patches, allowing it to capture complex global context and subtle manipulation artifacts. To ensure transparency, the system integrates Explainable AI (XAI) techniques, generating Attention Heatmaps to visually justify its classifications.

## 2. Introduction
Deepfake technology has evolved to a point where the human eye can no longer reliably distinguish between authentic and manipulated media. Traditional detection mechanisms often rely on localized pixel analysis, which sophisticated AI models can easily bypass. 

The primary objective of this project is to build a robust, web-based deepfake detection tool that not only provides a binary classification (`REAL` or `FAKE`) but also offers profound explainability. By adopting a Vision Transformer architecture, this system looks at the global structure and contextual relationships across the entire image, providing a superior detection mechanism.

## 3. System Architecture & Methodology
The core classification engine of De-Fake is built upon the Vision Transformer (ViT) paradigm. The end-to-end workflow is defined as follows:

### 3.1. Image Preprocessing
Input images uploaded via the web interface are asynchronously sent to the Flask backend. The image undergoes standard normalization and is resized to a strict `224x224` pixel resolution to satisfy the dimensional requirements of the ViT model.

### 3.2. Patch Embedding Extraction
Instead of analyzing the image pixel-by-pixel, the Vision Transformer partitions the `224x224` image into a fixed grid of `16x16` patches. This results in exactly 196 distinct patches (represented in a `14x14` visual grid on the frontend). Each patch is linearly projected into a one-dimensional vector, essentially treating the image patches similarly to how Natural Language Processing (NLP) models treat words in a sentence.

### 3.3. Positional Encoding and Self-Attention
Because transformers do not possess inherent knowledge of spatial geometry, Positional Encodings are injected into the patch vectors. The data is then processed through multiple Multi-Head Self-Attention layers. This mechanism allows the model to weigh the importance of different patches relative to one another, isolating inconsistencies often found in AI-generated imagery (e.g., asymmetrical lighting or distorted textures).

### 3.4. Classification
The final output sequence is pooled and passed through a Multi-Layer Perceptron (MLP) classification head. A Softmax function yields a probabilistic confidence score, categorizing the input strictly as either `FAKE` or `REAL`.

## 4. Explainable AI (XAI) Integration
A major challenge with Deep Learning models is their "black box" nature. To address this, De-Fake implements crucial transparency features:
* **Attention Heatmaps:** The system extracts the activation weights from the final layers of the neural network and overlays them onto the original image. The warmer colors (reds and yellows) indicate the precise physical locations where the model detected anomalies or synthetic artifacts.
* **Patch Visualization:** The UI dynamically renders the sequential slicing of the image, allowing the user to visualize the exact spatial distribution of the data being fed into the Transformer.

## 5. Technology Stack
The project leverages a modern, decoupled architecture:
* **Machine Learning Framework:** PyTorch & Torchvision (Model inference and tensor operations).
* **Backend Server:** Python with Flask (RESTful API endpoints handling image processing).
* **Computer Vision:** OpenCV and Pillow (PIL) for generating heatmaps and image manipulation.
* **Frontend Interface:** HTML5, CSS3, and JavaScript (Asynchronous requests and dynamic DOM updates).

## 6. Dataset and Training Context
The underlying model (`deepfake_vit.pth`) was trained on a comprehensive dataset consisting of thousands of real and synthetically generated images (such as the CIFAKE dataset). The training process utilized an 80/20 train-validation split to ensure the model generalizes well to unseen data and avoids overfitting. The `Data` directory provided in this repository contains a sample of these images to facilitate robust local testing.

## 7. Setup and Installation Guide
To deploy the system locally for demonstration or development purposes, follow the environment setup instructions:

**Step 1: Environment Preparation**
Ensure Python 3.8+ is installed. Navigate to the project root directory:
```bash
cd "C:\Users\pasula charan\Downloads\de fake\finalde-fake\De-Fake"
```

**Step 2: Dependency Installation**
Install all required libraries via pip:
```bash
pip install torch torchvision opencv-python Flask Pillow numpy
```

**Step 3: Server Initialization**
Start the Flask WSGI server:
```bash
python app.py
```

**Step 4: Application Access**
Launch a modern web browser and navigate to the local host address: `http://127.0.0.1:5000`

## 8. Usage Instructions
1. Initialize the local server using the instructions above.
2. Click on the designated upload zone on the web interface.
3. Navigate to the `Data/FAKE` or `Data/REAL` directories.
4. Select a test image and observe the real-time inference.
5. The system will display the final prediction, the structural patches, and the interpretative heatmap.

## 9. Conclusion
The **De-Fake** project successfully demonstrates a highly accurate and scalable approach to deepfake detection. By bridging advanced Vision Transformers with Explainable AI visualizations, the system not only acts as a classification tool but also serves as an educational platform to understand the vulnerabilities in synthetic media generation. Future iterations could involve deploying the model to a cloud architecture or integrating video frame-by-frame analysis.
