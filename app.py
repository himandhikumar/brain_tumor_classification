# app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import io

st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

model_file = st.file_uploader("Upload trained model (.pt)", type=["pt"])

@st.cache_resource
def load_model():
    model = torch.load("efficient_b0_best_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

model = load_model(model_file)

image_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if image_file and model:
    img = Image.open(image_file).convert('RGB')
    st.image(img, caption="Input MRI", use_column_width=True)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])

    input_tensor = transform(img).unsqueeze(0)
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()

    st.success(f"Prediction: {'Tumor' if pred else 'No Tumor'}")
elif image_file:
    st.warning("Please upload a model first.")
