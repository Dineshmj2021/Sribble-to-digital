import cv2
import numpy as np
import easyocr
import streamlit as st


@st.cache_resource
def get_ocr_reader():
    """
    Load EasyOCR only once.
    """
    return easyocr.Reader(['en'], gpu=False)


def enhance_image(image):
    """
    Improve handwritten image quality.
    """

    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image

    denoised = cv2.bilateralFilter(gray, 9, 75, 75)

    clahe = cv2.createCLAHE(
        clipLimit=2.0,
        tileGridSize=(8, 8)
    )
    enhanced = clahe.apply(denoised)

    kernel = np.array([
        [-1, -1, -1],
        [-1,  9, -1],
        [-1, -1, -1]
    ])

    sharpened = cv2.filter2D(
        enhanced,
        -1,
        kernel
    )

    return sharpened


def extract_text(image):
    """
    OCR with automatic rotation handling.
    """

    reader = get_ocr_reader()

    rotations = [
        image,
        cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE),
        cv2.rotate(image, cv2.ROTATE_180),
        cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    best_text = ""

    for img in rotations:
        try:
            results = reader.readtext(
                img,
                detail=0,
                paragraph=True
            )

            text = "\n".join(results)

            if len(text) > len(best_text):
                best_text = text

        except Exception:
            continue

    return best_text.strip()
