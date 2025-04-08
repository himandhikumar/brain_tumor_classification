import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="MRI Brain Tumor Classifier", layout="centered")
st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

@st.cache_resource
def load_model():
    # Load the architecture
    model = models.efficientnet_b0(weights=None)  # don't load pretrained weights here
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)  # Adjust for binary classification
    model.load_state_dict(torch.load("efficient_b0_best_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Image uploader
image_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if image_file:
    img = Image.open(image_file).convert("RGB")
    
    # Show image centered and smaller
    st.image(img, caption="Input MRI Image", use_column_width=False, width=300)

    # Transform image
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
        label = "Tumor" if prediction else "No Tumor"

    st.success(f"Prediction: **{label}**")
