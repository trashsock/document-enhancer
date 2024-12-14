import streamlit as st
import cv2
import numpy as np
from PIL import Image
import pytesseract

# Supported languages for OCR
language_options = {
    "English": "eng",
    "Spanish": "spa",
    "French": "fra",
    "German": "deu",
    "Italian": "ita",
    "Chinese (Simplified)": "chi_sim",
    "Chinese (Traditional)": "chi_tra",
    "Japanese": "jpn",
    "Hindi": "hin",
    "Arabic": "ara",
    "Russian": "rus",
    "Korean": "kor",
    "Bengali": "ben",
    "Tamil": "tam",
    "Urdu": "urd",
}

# Image enhancement function
def enhance_image(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    denoised = cv2.fastNlMeansDenoising(gray, None, 30, 7, 21)
    enhanced = cv2.adaptiveThreshold(denoised, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)
    return enhanced

# OCR function
def extract_text(image, lang):
    return pytesseract.image_to_string(image, lang=lang)

# Streamlit App
st.title("📄 Multi-Language Document Enhancement & OCR Tool")
st.write("Upload a scanned or low-quality document image. The tool enhances its readability and extracts text in your chosen language.")

# File uploader
uploaded_file = st.file_uploader("Upload an image file", type=["png", "jpg", "jpeg"])

# Language selection
language = st.selectbox("Select OCR Language", options=list(language_options.keys()), index=0)
selected_language_code = language_options[language]

if uploaded_file:
    # Read and display the uploaded image
    image = np.array(Image.open(uploaded_file))
    st.image(image, caption="Uploaded Image", use_column_width=True)

    # Enhance the image
    st.write("Enhancing the image...")
    enhanced_image = enhance_image(image)
    st.image(enhanced_image, caption="Enhanced Image", use_column_width=True, channels="GRAY")

    # Extract text using OCR
    st.write("Extracting text...")
    extracted_text = extract_text(enhanced_image, selected_language_code)
    st.text_area("Extracted Text", extracted_text, height=200)

    # Option to download the enhanced image
    save_button = st.download_button(
        label="Download Enhanced Image",
        data=cv2.imencode('.png', enhanced_image)[1].tobytes(),
        file_name="enhanced_image.png",
        mime="image/png"
    )
