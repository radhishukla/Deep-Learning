# Deep-Learning

This repo includes all the practicals from my Deep Learning lab, implemented using Python, TensorFlow, Keras, NumPy, and a sprinkle of patience. Dive in to explore neural networks from scratch, classification tasks, optimization experiments, LSTMs, and even some autoencoder wizardry. 🧙‍♂️📉

---

## 📂 Contents

### 1. 📉 Insurance Cost Prediction using ANN
- Performed EDA to understand the dataset.
- Built baseline models using:
  - Perceptron
  - Deep Neural Networks (DNNs)
- Played with hyperparameters:
  - Epochs
  - Batch Size
  - Learning Rates
- Optimizers tested:
  - SGD
  - Momentum
  - Nesterov Accelerated Gradient
- Evaluated using metrics like MSE, MAE.
- ✅ Saved the trained ANN model.
- 🔍 Comparative analysis of results included.

---

### 2. 🪷 Perceptron Classifier on Iris Dataset
- Used the classic Iris dataset 🌸.
- Focused on binary classification: `Setosa` vs `Versicolor`.
- Selected features: `Sepal Length` & `Sepal Width`.
- Implemented the **Perceptron Learning Algorithm** from scratch.
- Visualized the decision boundary.
- Accuracy evaluated on test set.

---

### 3. 🏗️ Feedforward Neural Network from Scratch
- **Task A:** Binary Classification using `make_moons` / `make_circles`.
  - Activation: Sigmoid
  - Loss: Mean Squared Error
- **Task B:** Multiclass Classification
  - Network: 4-3-4-3 (input to output)
  - Activation: Sigmoid (hidden), Softmax (output)
  - Loss: Binary Cross Entropy (yes, for multiclass—educational choice!)
- Built using only NumPy — no high-level libraries. We raw-coded it. 😤

---

### 4. 🧱 Multiclass Neural Net w/ ReLU + Softmax
- Architecture: 4 input → 3 hidden → 4 hidden → 3 output
- ReLU used in hidden layers, Softmax in output
- Loss: Binary Cross Entropy
- Implemented optimizers:
  - SGD
  - Momentum
  - NAG
- Evaluated and compared the performance.

---

### 5. 💉 Diabetes Prediction using ANN
- Performed thorough EDA.
- Trained ANN on diabetes dataset.
- Regularization Techniques applied:
  - L1, L2
  - Dropout
  - Early Stopping
- Tracked training using runtime chart ⏱️.
- Comparative analysis included.
- ✅ Model saved for future diagnosis 🤖

---

### 6. 🍓 CNN & Dense Backprop from Scratch + Fruit Classification
- **Part A:** Wrote custom forward and backward pass for a dense network. No cheating.
- **Part B:** Fruit Classification using CNN
  - Dataset: [`utkarshsaxenadn/fruits-classification`](https://www.kaggle.com/datasets/utkarshsaxenadn/fruits-classification)
  - Achieved 🍌 vs 🍎 vs 🍇 greatness.
  - Compared custom and pre-trained models.

---

### 7. 📈 LSTM for Stock Price Prediction (Google)
- Task: Predict next 15-day opening prices.
- Dataset preprocessed with 75/25 split, timestamp = 50.
- 4 stacked LSTM layers w/ 50 neurons each.
- Loss Function: Mean Squared Error
- Results visualized 📊 and plotted like real traders do (minus the losses).

---

### 8. 🔁 Autoencoder for MNIST Digit Regeneration
- Used 784 → 128 → 64 → 32 (encoder)
- Reconstructed back using decoder.
- Dataset: MNIST (handwritten digits)
- Model: Fully connected Autoencoder
- Output quality: Surprisingly crisp for a compressed code size of 32!
- Loss: Binary Cross Entropy

---

## 🛠️ Tools & Libraries Used
- Python 🐍
- TensorFlow / Keras
- NumPy / Pandas
- Matplotlib / Seaborn
- scikit-learn
- Jupyter Notebooks (aka code playgrounds)

---

## 📌 How to Run
```bash
git clone https://github.com/yourusername/deep-learning-practicals.git
cd deep-learning-practicals
# Open desired notebook and run using Jupyter/Colab
