# Animal-and-Object-Classification
# CIFAR-10 Image Classification using ResNet

A deep learning project that classifies images into 10 categories using a ResNet-style Convolutional Neural Network (CNN) built with TensorFlow and Keras.

The model was trained on the CIFAR-10 dataset and achieved **91.23% test accuracy**. The trained model is shared in this repository for inference and further experimentation.

The project was developed using Google Colab. A Streamlit web interface for image upload and prediction is planned as a future enhancement. The application is not currently deployed.

## 📌 Project Overview

Image classification is a computer vision task in which a deep learning model identifies the category of an image.

In this project, a ResNet-style CNN learns visual features from CIFAR-10 images and predicts one of 10 classes.

### Key Highlights

- Implemented a ResNet-style CNN with residual connections.
- Used TensorFlow and Keras for model development.
- Trained and evaluated the model using the CIFAR-10 dataset.
- Applied batch normalization, dropout, and data augmentation.
- Achieved 91.23% test accuracy.
- Saved the trained model in Keras format.
- Prepared the project for future integration with a Streamlit application.

## 🎯 Project Objectives

- Understand CNN-based image classification.
- Explore residual connections in deep neural networks.
- Train and evaluate a model using a standard image dataset.
- Measure classification performance on unseen test images.
- Share the trained model for reuse and experimentation.
- Prepare a foundation for a future interactive image-classification application.

## 🗂️ Dataset

**Dataset:** CIFAR-10

CIFAR-10 is a publicly available image dataset containing 60,000 color images distributed across 10 classes.

| Property | Details |
|---|---|
| Total images | 60,000 |
| Training images | 50,000 |
| Test images | 10,000 |
| Image dimensions | 32 × 32 pixels |
| Color channels | RGB |
| Number of classes | 10 |

### Image Classes

1. Airplane
2. Automobile
3. Bird
4. Cat
5. Deer
6. Dog
7. Frog
8. Horse
9. Ship
10. Truck

### Dataset Source

[CIFAR-10 Official Website — University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

The dataset can be loaded directly through TensorFlow:

```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

print("Training images:", x_train.shape)
print("Testing images:", x_test.shape)
```

TensorFlow downloads the dataset automatically if it is not already available in the environment.

## 🧠 Model Architecture

The project uses a ResNet-style CNN inspired by residual learning.

Unlike a basic CNN, residual blocks contain skip connections that allow information to bypass convolutional layers. These connections help gradients flow through deeper networks.

### Architecture Components

1. **Input layer:** Accepts RGB images of shape 32 × 32 × 3.
2. **Rescaling:** Normalizes pixel values to the range 0–1.
3. **Data augmentation:** Applies random horizontal flipping and translation during training.
4. **Initial convolution:** Extracts low-level visual features.
5. **Residual blocks:** Learn increasingly complex features using convolutional layers and skip connections.
6. **Batch normalization:** Helps stabilize training.
7. **Global average pooling:** Reduces feature maps to a compact representation.
8. **Dense layer:** Learns features useful for classification.
9. **Dropout:** Helps reduce overfitting.
10. **Softmax output:** Produces probabilities for the 10 classes.

### Training Configuration

| Parameter | Value |
|---|---|
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Evaluation metric | Accuracy |
| Batch size | 64 |
| Maximum epochs | 100 |
| Early stopping | Used |
| Learning-rate scheduling | ReduceLROnPlateau |

The actual number of training epochs may be lower than 100 because early stopping can terminate training when validation performance stops improving.

## 📊 Model Performance

The trained model achieved the following results on the CIFAR-10 test dataset.

| Metric | Result |
|---|---:|
| Test accuracy | 91.23% |
| Test loss | 0.3598 |
| Number of test images | 10,000 |
| Number of output classes | 10 |

### Results Interpretation

The model correctly classified approximately 9,123 out of 10,000 test images.

The test accuracy measures the proportion of correct predictions across the test dataset. The test loss measures the model's prediction error according to the selected loss function.

These results describe the reported evaluation of the trained model. Performance on arbitrary real-world images may differ because such images can have different backgrounds, viewpoints, resolutions, and visual characteristics.

## 🛠️ Technologies Used

- **Python** — Programming language
- **TensorFlow** — Deep learning framework
- **Keras** — Neural network API
- **NumPy** — Numerical computation
- **Matplotlib** — Training visualization
- **Scikit-learn** — Evaluation metrics and confusion matrix
- **Google Colab** — Model training environment
- **Git and GitHub** — Version control and project sharing

## 📁 Repository Structure

```text
cifar10-resnet-image-classifier/
│
├── cifar10_resnet_91.keras
├── README.md
├── requirements.txt
└── training.ipynb
```

| File | Description |
|---|---|
| `cifar10_resnet_91.keras` | Trained ResNet-style CNN model |
| `README.md` | Project documentation |
| `requirements.txt` | Python dependencies |
| `training.ipynb` | Training notebook, if included in the repository |

The model file must be available locally before running the inference examples below. The notebook is optional but recommended for understanding and reproducing the training process.

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd cifar10-resnet-image-classifier
```

Replace `<YOUR_GITHUB_REPOSITORY_URL>` with the actual URL of your GitHub repository.

### 2. Create a virtual environment

**Windows:**

```powershell
python -m venv .venv
.venv\Scripts\activate
```

**Linux/macOS:**

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Use a TensorFlow version compatible with the version used to save the model.

### 4. Load the trained model

Create a Python file named `predict.py` and use the following code:

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "cifar10_resnet_91.keras",
    compile=False
)

print("Model loaded successfully!")
model.summary()
```

Run it:

```bash
python predict.py
```

If the model loads successfully, you can proceed to image prediction.

## 🔍 Making Predictions

The following example demonstrates how to predict a CIFAR-10 test image using the saved model.

```python
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Load test data
(_, _), (x_test, y_test) = cifar10.load_data()

# Load trained model
model = tf.keras.models.load_model(
    "cifar10_resnet_91.keras",
    compile=False
)

# Select one test image
index = 25
image = x_test[index]

# Predict
probabilities = model.predict(
    image.astype("float32")[None, ...],
    verbose=0
)[0]

predicted_index = int(np.argmax(probabilities))

print("Predicted class:", class_names[predicted_index])
print("Actual class:", class_names[int(y_test[index][0])])
print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")

plt.imshow(image)
plt.title(f"Predicted: {class_names[predicted_index]}")
plt.axis("off")
plt.show()
```

The input image is passed in its original 0–255 pixel range because the model already contains a rescaling layer.

## 🚧 Current Project Status

- [x] CIFAR-10 dataset loaded
- [x] ResNet-style CNN implemented
- [x] Model trained in Google Colab
- [x] Model evaluated on the test dataset
- [x] 91.23% test accuracy achieved
- [x] Trained model shared on GitHub
- [ ] Streamlit image-upload interface
- [ ] Local application testing
- [ ] Online deployment

The current deliverable is the trained image-classification model. The web application and deployment are planned future enhancements.

## 🔮 Future Improvements

- Develop a Streamlit application for image upload and classification.
- Add a confusion matrix and class-wise precision, recall, and F1-score.
- Implement Grad-CAM to visualize important image regions.
- Experiment with stronger data augmentation and alternative architectures.
- Add batch prediction for multiple images.
- Optimize model inference and deployment size.
- Deploy the application using a suitable hosting platform.

## ⚠️ Limitations

- The model is trained to recognize only the 10 CIFAR-10 classes.
- CIFAR-10 images are small, with a resolution of 32 × 32 pixels.
- Resizing high-resolution photographs can remove useful details.
- Predictions on images outside the CIFAR-10 distribution may be unreliable.
- Confidence scores are not guarantees of correctness.
- Test accuracy does not necessarily represent performance on real-world photographs.

## 👨‍💻 Author

**Suprakash Ghosh**

B.Tech in Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

## 📄 License

This project is intended for educational and research purposes.

If you plan to distribute the project publicly, add an appropriate `LICENSE` file and verify the applicable licenses for dependencies, datasets, and other redistributed assets.

---

⭐ If you find this project useful, consider starring the repository on GitHub.
