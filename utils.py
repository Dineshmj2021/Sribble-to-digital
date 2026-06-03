import cv2
import numpy as np
import easyocr

def get_ocr_reader():
    """Initializes EasyOCR reader once."""
    # Use CPU by default to avoid CUDA issues in limited environments
    return easyocr.Reader(['en'], gpu=False)

def enhance_image(image):
    """Preprocesses image for better OCR results without over-saturating."""
    # Convert to grayscale if needed
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image
        
    # 1. Denoising while preserving edges (Bilateral Filter)
    # Better for handwriting than NlMeans
    denoised = cv2.bilateralFilter(gray, 9, 75, 75)
    
    # 2. Contrast Enhancement (CLAHE)
    # This avoids the "over-saturated" look of simple thresholding
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    enhanced = clahe.apply(denoised)
    
    # 3. Subtle Sharpening
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(enhanced, -1, kernel)
    
    return sharpened

def extract_text(image):
    """Uses EasyOCR to detect English text."""
    try:
        reader = get_ocr_reader()
        results = reader.readtext(image)
        
        # Combine detected words into a single string
        text_lines = [res[1] for res in results]
        return " ".join(text_lines)
    except Exception as e:
        print(f"OCR Error: {e}")
        return ""