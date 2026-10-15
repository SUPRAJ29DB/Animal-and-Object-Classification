# Animal-and-Object-Classification
# CIFAR-10 Image Classification Using ResNet

A TensorFlow/Keras ResNet-style Convolutional Neural Network (CNN) that classifies 32 × 32 RGB images into the 10 CIFAR-10 categories.

The trained model reportedly achieved **91.23% test accuracy** on CIFAR-10. The model is shared on GitHub for inference and further experimentation. A Streamlit application is planned but is not currently deployed.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-orange)](https://www.tensorflow.org/)
[![Dataset](https://img.shields.io/badge/Dataset-CIFAR--10-green)](https://www.cs.toronto.edu/~kriz/cifar.html)

## Features

- ResNet-style CNN with residual connections
- Batch normalization and dropout
- Random horizontal flipping and translation augmentation
- 10-class image classification
- Saved Keras model for inference
- Training and evaluation performed using Google Colab
- Streamlit integration planned for future development

## Model Performance

| Metric | Result |
|---|---:|
| Dataset | CIFAR-10 |
| Test images | 10,000 |
| Reported test accuracy | 91.23% |
| Reported test loss | 0.3598 |
| Number of classes | 10 |

These are the results recorded during the model evaluation. The training notebook and evaluation code should be included in the repository to help others reproduce them.

## Dataset

This project uses the CIFAR-10 dataset, containing 60,000 color images divided into 10 classes.

| Split | Number of images |
|---|---:|
| Training | 50,000 |
| Testing | 10,000 |
| Image dimensions | 32 × 32 × 3 |

### Supported classes

`airplane`, `automobile`, `bird`, `cat`, `deer`, `dog`, `frog`, `horse`, `ship`, `truck`

Dataset source: [CIFAR-10 — University of Toronto](https://www.cs.toronto.edu/~kriz/cifar.html)

Load the dataset using TensorFlow:

```python
from tensorflow.keras.datasets import cifar10

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
```

## Model Architecture

The model uses residual blocks to learn increasingly complex image features while allowing information to flow through skip connections.

### Simplified architecture

```text
Input: 32 × 32 × 3
        |
Rescaling (0–1)
        |
Data Augmentation
        |
Initial 3 × 3 Convolution
        |
Residual Block — 32 filters
        |
Residual Block — 32 filters
        |
Residual Block — 64 filters
        |
Residual Block — 64 filters
        |
Residual Block — 128 filters
        |
Residual Block — 128 filters
        |
Global Average Pooling
        |
Dense Layer — 128 units
        |
Dropout
        |
Softmax — 10 classes
```

### Training configuration

| Parameter | Configuration |
|---|---|
| Framework | TensorFlow / Keras |
| Optimizer | Adam |
| Initial learning rate | 0.001 |
| Loss function | Sparse categorical cross-entropy |
| Batch size | 64 |
| Maximum epochs | 100 |
| Early stopping | Enabled |
| Learning-rate scheduling | ReduceLROnPlateau |

The actual training duration and stopping epoch depend on validation performance and the available hardware.

## Quick Start

### 1. Clone the repository

```bash
git clone https://github.com/SUPRAJ29DB/Animal-and-Object-Classification.git
cd Animal-and-Object-Classification
```

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

Use a TensorFlow version compatible with the version used to save the model. Exact tested versions should be documented in `requirements.txt`.

### 4. Load the trained model

Ensure `cifar10_resnet_91.keras` is present in the repository directory.

```python
import tensorflow as tf

model = tf.keras.models.load_model(
    "cifar10_resnet_91.keras",
    compile=False
)

model.summary()
```

The model can now be used for image inference.

## Prediction Example

The following example predicts the class of an image from the CIFAR-10 test set.

```python
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

from tensorflow.keras.datasets import cifar10

class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

(_, _), (x_test, y_test) = cifar10.load_data()

model = tf.keras.models.load_model(
    "cifar10_resnet_91.keras",
    compile=False
)

index = 25
image = x_test[index]

probabilities = model.predict(
    image.astype("float32")[None, ...],
    verbose=0
)[0]

predicted_index = int(np.argmax(probabilities))

print("Predicted:", class_names[predicted_index])
print("Actual:", class_names[int(y_test[index, 0])])
print(f"Confidence: {probabilities[predicted_index] * 100:.2f}%")

plt.imshow(image)
plt.title(f"Predicted: {class_names[predicted_index]}")
plt.axis("off")
plt.show()
```

The model includes a rescaling layer, so input images should retain pixel values in the 0–255 range.

## Repository Structure

```text
Animal-and-Object-Classification/
├── cifar10_resnet_91.keras
├── README.md
├── requirements.txt
├── training.ipynb
└── assets/
```

- `cifar10_resnet_91.keras` — trained model
- `README.md` — project documentation
- `requirements.txt` — Python dependencies
- `training.ipynb` — training notebook, if included
- `assets/` — optional screenshots and result visualizations

Only include files and directories that actually exist in the repository.

## Reproducibility

For reproducibility, include the training notebook with the following information:

- Dataset loading and preprocessing
- Model architecture
- Training configuration
- Validation and test evaluation
- Accuracy and loss graphs
- Confusion matrix and classification report

The saved model supports inference, while the notebook documents the training process.

## Limitations

- The classifier recognizes only the 10 CIFAR-10 categories.
- CIFAR-10 images have a resolution of just 32 × 32 pixels.
- Resizing larger images may remove important visual details.
- Performance on arbitrary real-world photographs may be lower because of differences in backgrounds, viewpoints, lighting conditions, image resolutions, and object appearances.
- Confidence scores indicate model probabilities, not guarantees of correctness.

## Future Improvements

- Build a Streamlit interface for image uploads and predictions.
- Add a confusion matrix and per-class evaluation metrics.
- Visualize model attention using Grad-CAM.
- Experiment with stronger data augmentation and alternative architectures.
- Add batch prediction support.
- Deploy the application online when ready.

## Current Project Status

- [x] Dataset loaded and prepared
- [x] ResNet-style CNN implemented
- [x] Model trained in Google Colab
- [x] Model evaluated
- [x] Trained model shared on GitHub
- [ ] Training notebook published, if not already uploaded
- [ ] Streamlit interface completed
- [ ] Application deployed

## Author

**Suprakash Ghosh**

B.Tech — Computer Science and Engineering  
Specialization: Artificial Intelligence and Machine Learning

## License

Add a `LICENSE` file before declaring this project open source. If you choose the MIT License, ensure that it applies to the code you own and separately check the licensing terms for the dataset, trained model, and third-party assets.

---


⭐ If you find this project useful, consider starring the repository on GitHub.
