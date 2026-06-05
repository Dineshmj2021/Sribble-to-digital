import streamlit as st
import cv2
import numpy as np
from PIL import Image
import os
from utils import enhance_image, extract_text, get_ocr_reader
from google import genai
from google.genai import types

# Page Config
# Page Config
st.set_page_config(page_title="Scribble to Digital", layout="wide")

st.title("Scribble to Digital")

# Load OCR model
get_ocr_reader()

# Sidebar for API Key (Fallback if env var not set)
if "GEMINI_API_KEY" not in st.secrets:
    st.error("GEMINI_API_KEY not found in Streamlit Secrets.")
    st.stop()

api_key = st.secrets["GEMINI_API_KEY"]
client = genai.Client(api_key=api_key)

uploaded_file = st.file_uploader("Upload handwritten image", type=["jpg", "png", "jpeg"])

if uploaded_file:
    # Load image
    image = Image.open(uploaded_file)
    # Resize large mobile photos
    image.thumbnail((1200, 1200))
    img_array = np.array(image)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Uploaded Image")
        st.image(image,width="stretch")
        
    with col2:
        st.subheader("Processed Image")
        processed_img = enhance_image(img_array)
        st.image(processed_img, width="stretch")
        
    if st.button("Convert to Digital"):
        with st.spinner("Processing OCR and AI cleanup..."):
            try:
                # OCR Step
                raw_text = extract_text(processed_img)
                
                if not raw_text.strip():
                    st.warning("EasyOCR could not detect text. Falling back to Gemini Vision OCR...")
                    raw_text = "No text detected by EasyOCR, using Gemini Vision directly."

                st.subheader("Raw OCR Text")
                st.text_area("Detected text", raw_text, height=150)
                
                # Gemini Step (Multimodal Fallback)
                # We pass the image directly to Gemini as it's much better at handwriting
                # than EasyOCR. This ensures we always get a correct digital output.
                
                # Convert processed image back to bytes for Gemini
                _, buffer = cv2.imencode('.jpg', processed_img)
                img_bytes = buffer.tobytes()
                
                prompt = f"""
                I have the following raw OCR text from handwritten notes: "{raw_text}". 
                Please correct any OCR mistakes, fix spelling, reconstruct readable sentences, and extract actionable tasks.
                
                If the raw text provided is poor or empty, please use the attached image to perform your own OCR and extract the information.
                
                Format your response EXACTLY as follows:
                Clean Notes:
                * [note item]
                
                To-Do List:
                * [task item]
                """
                
                response = client.models.generate_content(
                    model="gemini-2.5-flash-lite",
                    contents=[
                        types.Part.from_bytes(data=img_bytes, mime_type="image/jpeg"),
                        prompt
                    ]
                )
                
                st.subheader("Digital Output")
                if hasattr(response, "text") and response.text:
                    st.markdown(response.text)
                else:
                    st.warning("No response returned from Gemini.")
                
            except Exception as e:
                st.exception(e)
else:
    st.info("Please upload an image to get started.")
