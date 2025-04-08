#app.py
import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image

st.title("🧠 MRI Brain Tumor Classifier - EfficientNetB0")

@st.cache_resource
def load_model():
    model = torch.load("efficient_b0_best_model.pt", map_location=torch.device("cpu"))
    model.eval()
    return model

model = load_model()

image_file = st.file_uploader("Upload an MRI image", type=["jpg", "jpeg", "png"])

if image_file:
    img = Image.open(image_file).convert('RGB')

    # Resize for preview
    preview_img = img.resize((150, 150))

    # Center the image using HTML
    st.markdown(
        f"<div style='text-align: center;'><img src='data:image/png;base64,{st.image(preview_img, use_column_width=False).image_to_base64()}' width='150'></div>",
        unsafe_allow_html=True
    )

    if model:
        # Preprocess for model
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

        input_tensor = transform(img).unsqueeze(0)
        with torch.no_grad():
            output = model(input_tensor)
            pred = torch.argmax(output, 1).item()

        st.success(f"Prediction: {'Tumor' if pred else 'No Tumor'}")
    else:
        st.warning("Please upload the model file.")
