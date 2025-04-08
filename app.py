import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image

st.set_page_config(page_title="MRI Brain Tumor Classifier", layout="centered")
st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

@st.cache_resource
def load_model():
    from torchvision import models
    import torch.nn as nn

    # Load base EfficientNetB0 model
    model = models.efficientnet_b0(weights=None)

    # Modify classifier layer to match your binary classification task
    model.classifier[1] = nn.Linear(model.classifier[1].in_features, 2)

    # Load trained weights
    state_dict = torch.load("efficient_b0_best_model.pt", map_location="cpu")
    model.load_state_dict(state_dict)
    model.eval()

    return model


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
    device = torch.device("cpu")  # or "cuda" if running locally on GPU
    input_tensor = input_tensor.to(device)
    model.to(device)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()
        label = "Tumor" if prediction else "No Tumor"

    st.success(f"Prediction: **{label}**")
