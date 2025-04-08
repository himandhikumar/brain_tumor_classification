import streamlit as st
import torch
from torchvision import models, transforms
from PIL import Image
import io

st.set_page_config(page_title="MRI Brain Tumor Classifier", layout="centered")
st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

@st.cache_resource
def load_model():
    model = models.efficientnet_b0(pretrained=False)
    model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, 2)
    model.load_state_dict(torch.load("efficient_b0_best_model.pt", map_location="cpu"))
    model.eval()
    return model

model = load_model()

# Upload image
image_file = st.file_uploader("📤 Upload an MRI image", type=["jpg", "jpeg", "png"])

if image_file:
    img = Image.open(image_file).convert('RGB')

    # Center the image and make it smaller
    st.markdown("<div style='text-align: center;'>", unsafe_allow_html=True)
    st.image(img, caption="🖼️ Uploaded MRI Image", width=300)
    st.markdown("</div>", unsafe_allow_html=True)

    # Image preprocessing
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor()
    ])
    input_tensor = transform(img).unsqueeze(0)

    # Predict
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, 1).item()

    st.success(f"✅ Prediction: **{'Tumor' if pred == 1 else 'No Tumor'}**")
