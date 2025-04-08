import streamlit as st
import torch
from torch import nn
from torchvision import models, transforms
from PIL import Image
import io

st.set_page_config(page_title="MRI Brain Tumor Classifier", layout="centered")
st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    for param in model.parameters():
        param.requires_grad = False
    model.classifier = nn.Sequential(
        nn.Linear(model.classifier[1].in_features, 1280),
        nn.ReLU(),
        nn.Dropout(0.4),
        nn.Linear(1280, 2)  # Adjust if you have different number of classes
    )
    model.load_state_dict(torch.load("efficient_b0_best_model.pt", map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

image_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if image_file and model:
    img = Image.open(image_file).convert('RGB')
    
    # Resize image for display
    display_img = img.copy()
    display_img.thumbnail((300, 300))  # Resize preview image

    # Center the image
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(display_img, caption="Input MRI", use_column_width=True)

    # Preprocess
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)

    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()

    result = "🧠 **Tumor Detected**" if pred == 1 else "✅ **No Tumor Detected**"

    st.markdown("---")
    st.markdown(f"<div style='text-align:center; font-size: 24px; color: #4CAF50;'>{result}</div>", unsafe_allow_html=True)
elif image_file and not model:
    st.warning("Please upload the model file first.")
