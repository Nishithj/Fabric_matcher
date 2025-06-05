import streamlit as st
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from sklearn.metrics.pairwise import cosine_similarity
import os
import numpy as np

# Set device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load pretrained ResNet34 model (feature extractor)
@st.cache_resource
def load_model():
    resnet34 = models.resnet34(pretrained=True)
    model = nn.Sequential(*list(resnet34.children())[:-1])
    model.eval()
    model.to(device)
    return model

model = load_model()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406],
                         [0.229, 0.224, 0.225])
])

# Extract feature from image
def extract_feature(image):
    image = image.convert("RGB")
    image = transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        feature = model(image).cpu().numpy().squeeze()
    return feature

# Load reference image features
@st.cache_data
def load_reference_features(train_folder="training"):
    features = {}
    for filename in os.listdir(train_folder):
        if filename.lower().endswith(('.jpg', '.png')):
            pattern_num = os.path.splitext(filename)[0]
            path = os.path.join(train_folder, filename)
            feat = extract_feature(Image.open(path))
            features[pattern_num] = feat
    return features

# Matcher
def match_image(test_image, reference_features, train_folder="training"):
    test_feat = extract_feature(test_image)
    best_match, max_sim = None, -1

    for pattern, ref_feat in reference_features.items():
        sim = cosine_similarity([test_feat], [ref_feat])[0][0]
        if sim > max_sim:
            max_sim = sim
            best_match = pattern

    matched_img_path = os.path.join(train_folder, f"{best_match}.jpg")
    matched_img = Image.open(matched_img_path)
    return best_match, max_sim, matched_img

# Streamlit App
st.title("🧵 Fabric Pattern Matcher")
st.write("Upload a fabric image or click one with your camera to find the matching pattern.")

uploaded_file = st.file_uploader("Upload or take a fabric image:", type=["jpg", "png"], accept_multiple_files=False, label_visibility="visible")

# Load reference features once
reference_features = load_reference_features("training")

if uploaded_file:
    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_column_width=True)

    with st.spinner("Matching pattern..."):
        pattern, similarity, matched_img = match_image(image, reference_features)

    st.success(f"🎯 Matched Pattern: {pattern} (Similarity: {similarity:.4f})")
    st.image(matched_img, caption=f"Matched Pattern: {pattern}", use_column_width=True)
