import streamlit as st
from PIL import Image
import torch
from torch import nn
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
from model import CustomCNN

# List of Imagenette class names
class_names = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute"
]

# Load the model once
@st.cache_resource
def load_model():
    model = CustomCNN(num_classes=len(class_names))
    model.load_state_dict(torch.load("best_model.pth", map_location=torch.device('cpu')))
    model.eval()
    return model

# Image preprocessing (same as your test_transform)
transform = transforms.Compose([
    transforms.Resize((128, 128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=(0.485, 0.456, 0.406),
                         std=(0.229, 0.224, 0.225))
])

# Streamlit app
st.title("Imagenette Image Classifier")

uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    input_tensor = transform(image).unsqueeze(0)  # [1, 3, 128, 128]

    model = load_model()
    with torch.no_grad():
        output = model(input_tensor)
        probabilities = torch.softmax(output[0], dim=0).numpy()

    predicted_index = np.argmax(probabilities)
    predicted_class = class_names[predicted_index]
    confidence = probabilities[predicted_index]

    st.subheader(f"Predicted Class: {predicted_class} ({confidence * 100:.2f}%)")

    st.subheader("Class Probabilities")
    fig, ax = plt.subplots()
    ax.barh(class_names, probabilities)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()
    st.pyplot(fig)
