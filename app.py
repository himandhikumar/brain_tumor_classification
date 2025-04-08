import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
import os

# Center title
st.markdown("<h1 style='text-align: center;'>🧠 MRI Brain Tumor Classifier</h1>", unsafe_allow_html=True)

# Load model
@st.cache_resource
def load_model():
    model = torch.load("efficient_b0_best_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

# Image uploader
image_file = st.file_uploader("Upload an MRI Image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])

# Prediction logic
if image_file is not None:
    img = Image.open(image_file).convert("RGB")

    # Show smaller centered image
    st.markdown("<h4 style='text-align: center;'>Uploaded MRI Image</h4>", unsafe_allow_html=True)
    st.image(img, width=250)

    # Preprocess image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0).to(torch.device("cpu"))

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()

    result = "🟢 Tumor Detected" if pred == 1 else "🟡 No Tumor Detected"
    st.markdown(f"<h3 style='text-align: center; color: green;'>{result}</h3>", unsafe_allow_html=True)
