# Handwritten Character Recognition using CNN

A deep learning project that recognizes handwritten digits (0–9) using a Convolutional Neural Network (CNN) trained on the **MNIST dataset**. It includes a training and evaluation pipeline and an interactive Streamlit app for uploading a handwritten digit image and viewing the prediction.

## Objective

Identify handwritten digits from input images using image preprocessing and deep learning.

## Features

- Train a CNN on the MNIST handwritten digit dataset.
- Recognize individual handwritten digits from 0 to 9.
- Preprocess uploaded images into a 28 × 28 grayscale format.
- Display the predicted digit and class probabilities.
- Evaluate the model on the MNIST test dataset.
- Generate accuracy/loss curves and a confusion matrix.

## Technologies

- Python
- TensorFlow / Keras
- NumPy
- Matplotlib
- Scikit-learn
- Pillow
- Streamlit
- MNIST dataset

## Project Structure

```text
Handwritten-Character-Recognition-using-CNN/
├── app.py
├── train.py
├── models/                 # Generated model and evaluation plots
├── requirements.txt
├── README.md
└── .gitignore
```

## Setup and Installation

### 1. Clone the repository

```bash
git clone https://github.com/subhangi2006/Handwritten-Character-Recognition-using-CNN.git
cd Handwritten-Character-Recognition-using-CNN
```

### 2. Create a virtual environment (recommended)

Windows:

```bash
python -m venv .venv
.venv\\Scripts\\activate
```

macOS / Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install dependencies

```bash
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Train the CNN

Run:

```bash
python train.py
```

The script downloads MNIST through TensorFlow/Keras, preprocesses the images, trains the CNN, evaluates it on the test dataset, and saves the best model and evaluation plots in the `models/` directory.

Generated files:

- `models/handwritten_digit_cnn.keras`
- `models/training_history.png`
- `models/confusion_matrix.png`

The model and plots are generated locally and are excluded from Git by `.gitignore`.

## Run the Streamlit Application

After training completes:

```bash
streamlit run app.py
```

Open the local URL shown in the terminal. Upload a clear image containing one handwritten digit (0–9) to view the prediction and class probabilities.

## CNN Architecture

The network includes:

1. Convolutional layers to learn visual features.
2. Batch normalization to help stabilize training.
3. Max pooling to reduce spatial dimensions.
4. Dropout layers to help reduce overfitting.
5. Dense layers for classification.
6. A 10-unit softmax output layer for digits 0–9.

## Evaluation

The training script reports test loss, test accuracy, precision, recall, and F1-score. It also saves accuracy/loss curves and a confusion matrix. Record the actual metrics after running the training script; no performance result is assumed in advance.

## Limitations

- The current model recognizes one digit per image, not complete words or sentences.
- Blurry, rotated, unusually styled, or multi-digit images may be classified incorrectly.
- Prediction confidence is not a guarantee of correctness.

## Future Scope

- Extend recognition to letters using the EMNIST dataset.
- Improve image preprocessing and digit centering.
- Explore word and sentence recognition using sequence models such as CRNN.

## Author

**Subhangi Panda**

GitHub: [@subhangi2006](https://github.com/subhangi2006)
