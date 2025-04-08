# 🧠 MRI Brain Tumor Classifier using EfficientNetB0

A deep learning-powered web app to classify brain MRI images as **Tumor** or **No Tumor** using EfficientNetB0. Built with PyTorch and Streamlit.

---

## 📊 Model Performance Comparison

| Model             | Loss     | Accuracy |
|------------------|----------|----------|
| Linear_Model      | 0.715871 | 0.392270 |
| Non_Linear_Model  | 0.786749 | 0.712171 |
| CNN_Model         | 0.665969 | 0.712171 |
| ResNet-18         | 0.358032 | 0.853618 |
| ResNet-50         | 0.348577 | 0.827303 |
| EfficientNetB0    | 0.203366 | 0.895559 |
| EfficientNet-B3   | 0.690284 | 0.601974 |

✅ **EfficientNetB0** performed best in terms of accuracy with relatively low loss.

---

## 🔧 Activities Performed

### 🧪 Data Preparation
- Preprocessed MRI images: Resized, normalized, and converted to tensors.
- Split dataset into training, validation, and test sets.

### 🧠 Model Building
- Implemented and trained various models:
  - Baseline Linear and Non-linear Neural Nets
  - Basic CNN
  - Pretrained ResNet18 and ResNet50
  - EfficientNetB0 and EfficientNetB3

### 📉 Training & Evaluation
- Fine-tuned pretrained models using Transfer Learning.
- Tracked performance metrics: Loss, Accuracy.
- Selected best-performing model (EfficientNetB0).

### 🌐 Web App Deployment
- Built a user-friendly Streamlit app:
  - Upload MRI image
  - Get tumor prediction + confidence score
  - See progress visually
- Deployed app on Streamlit Cloud.

🔗 **Live App:** [Click to Launch MRI Tumor Classifier](https://braintumorclassificationbyhimandhikumar.streamlit.app)

---

## 🚀 Tech Stack

- **Python**
- **PyTorch**
- **Torchvision**
- **Streamlit**
- **PIL** (Image processing)

---

## 🧠 Usage Instructions

1. Upload an MRI image (JPG/PNG).
2. Model predicts if it's `Tumor` or `No Tumor`.
3. Confidence score is displayed along with prediction.

---

## 📁 File Structure

```
├── app.py                     # Streamlit app script
├── efficient_b0_best_model.pt # Saved model weights
├── README.md                  # Project documentation
└── requirements.txt           # Python dependencies
```

---

## 📌 Future Work

- Add Grad-CAM for model interpretability
- Extend to multi-class tumor classification
- Deploy on Hugging Face Spaces

---

## 🤝 Let's Connect
If you liked this project or have suggestions, feel free to reach out or raise an issue.

---

> *Built with ❤️ for AI in Healthcare.*

