# 🗑️ Garbage Classification using Deep Learning

## 📌 Overview

This project focuses on building an intelligent image classification system to categorize garbage into **12 different classes** using deep learning.

The system leverages:

* **Transfer Learning (ResNet18)**
* **Custom CNN with Residual Blocks**
* **Robust evaluation metrics (Accuracy, Precision, Recall, F1 Score)**

---

## 📂 Dataset

The dataset consists of garbage images categorized into 12 classes such as:

* Battery
* Biological
* Brown Glass
* Cardboard
* Clothes
* Green Glass
* Metal
* Paper
* Plastic
* Shoes
* Trash
* White Glass

### 📁 Structure

```
data/
│
├── garbage_classification/              # Raw dataset
├── garbage_classification_processed/   # Split dataset
    ├── train/
    ├── val/
    ├── test/
```

---

## ⚙️ Features

* ✅ Modular project structure (industry-standard)
* ✅ Data preprocessing & augmentation
* ✅ ResNet18 (pretrained) implementation
* ✅ Custom CNN with Residual Blocks
* ✅ Training & validation pipeline
* ✅ Logging system
* ✅ Evaluation metrics:

  * Accuracy
  * Precision
  * Recall
  * F1 Score
* ✅ Confusion Matrix visualization
* ✅ Model saving (best model based on F1 score)
* ✅ Inference on:

  * Full test dataset
  * Single image input

---

## 🧠 Model Performance

| Metric   | Value |
| -------- | ----- |
| Accuracy | ~91%  |
| F1 Score | ~0.88 |

> Model selected based on best F1 score to handle class imbalance.

---

## 🚀 Installation

```bash
git clone <your-repo-link>
cd garbage-classification
pip install -r requirements.txt
```

---

## ▶️ Usage

### 1️⃣ Split Dataset

```bash
python split_dataset.py
```

---

### 2️⃣ Train Model

```bash
python train.py
```

---

### 3️⃣ Evaluate Model (Confusion Matrix)

```bash
python evaluate_model.py
```

---

### 4️⃣ Predict Single Image

```bash
python predict.py
```

---

## 📊 Confusion Matrix

After evaluation, the confusion matrix is saved at:

```
outputs/confusion_matrix_test.png
```

You can visualize model performance across all classes.

---

## 📁 Project Structure

```
garbage-classification/
│
├── data/
├── notebooks/
├── src/
│   ├── config/
│   ├── data/
│   ├── models/
│   ├── engine/
│   ├── utils/
│
├── outputs/
│   ├── models/
│   ├── logs/
│
├── train.py
├── evaluate_model.py
├── predict.py
├── split_dataset.py
├── requirements.txt
└── README.md
```

---

## 🧪 Technologies Used

* Python 3.10
* PyTorch
* Torchvision
* NumPy
* Scikit-learn
* Matplotlib / Seaborn

---

## 📈 Future Improvements

* 🔹 Hyperparameter tuning
* 🔹 Class imbalance handling (weighted loss)
* 🔹 Deployment with FastAPI
* 🔹 Streamlit UI for real-time predictions
* 🔹 Model optimization (ONNX / TensorRT)

---

## 🤝 Contribution

Contributions are welcome! Feel free to fork and improve.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

**Vajresh D. Kulkarni (Vajubhai)**
AI & ML Engineer 🚀
