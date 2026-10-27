
import re
import hashlib
from pathlib import Path

import streamlit as st
import numpy as np
import cv2
import easyocr
import tensorflow as tf
from PIL import Image

# Optional online medicine information
import requests


# =========================================================
# 1. APP CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="AI-Based Counterfeit Medicine Detection",
    page_icon="💊",
    layout="wide"
)

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "efficientnetb2_final.keras"

IMAGE_SIZE = 260


# =========================================================
# 2. MEDICINE INFORMATION DATABASE
# =========================================================

MEDICINE_INFO = {
    "paracetamol": {
        "generic": "Paracetamol",
        "use": "Commonly used to relieve pain and reduce fever."
    },
    "acetaminophen": {
        "generic": "Acetaminophen",
        "use": "Commonly used to relieve pain and reduce fever."
    },
    "dolo 650": {
        "generic": "Paracetamol",
        "use": "Used to relieve pain and reduce fever."
    },
    "crocin": {
        "generic": "Paracetamol (check the package)",
        "use": "Some Crocin products are used for pain and fever."
    },
    "calpol": {
        "generic": "Paracetamol (check the package)",
        "use": "Some Calpol products are used for pain and fever."
    },
    "ibuprofen": {
        "generic": "Ibuprofen",
        "use": "A medicine used to relieve pain, inflammation and fever."
    },
    "cetirizine": {
        "generic": "Cetirizine",
        "use": "An antihistamine used to relieve allergy symptoms."
    },
    "aspirin": {
        "generic": "Aspirin",
        "use": "Used for pain and fever; some formulations have other uses."
    },
    "ecosprin": {
        "generic": "Aspirin (verify the exact product)",
        "use": "Some products are used under medical guidance to prevent blood clots."
    },
    "ecosprin av": {
        "generic": "Combination product (verify the package)",
        "use": "Check the exact ingredients and consult a pharmacist."
    },
    "combiflam": {
        "generic": "Combination pain-relief medicine",
        "use": "Check the package for its exact ingredients and warnings."
    },
    "amoxicillin": {
        "generic": "Amoxicillin",
        "use": "An antibiotic used for certain bacterial infections when prescribed."
    },
    "augmentin": {
        "generic": "Amoxicillin and clavulanic acid (verify the product)",
        "use": "An antibiotic combination used for certain bacterial infections."
    },
    "azithromycin": {
        "generic": "Azithromycin",
        "use": "An antibiotic used for certain bacterial infections when prescribed."
    },
    "azithral": {
        "generic": "Azithromycin (verify the package)",
        "use": "An antibiotic brand; use only as prescribed."
    },
    "metformin": {
        "generic": "Metformin",
        "use": "Used to help manage type 2 diabetes."
    },
    "omeprazole": {
        "generic": "Omeprazole",
        "use": "Reduces stomach acid."
    },
    "pantoprazole": {
        "generic": "Pantoprazole",
        "use": "Reduces stomach acid."
    },
    "diclofenac": {
        "generic": "Diclofenac",
        "use": "Used to relieve pain and inflammation."
    }
}


# General, non-exhaustive side-effect summaries for common medicines.
# These are educational summaries; exact risks depend on ingredients, dose,
# age, medical history, and the product label.
COMMON_SIDE_EFFECTS = {
    "paracetamol": "Usually well tolerated at recommended doses. Overdose can cause severe liver damage; allergic reactions are possible.",
    "acetaminophen": "Usually well tolerated at recommended doses. Overdose can cause severe liver damage; allergic reactions are possible.",
    "dolo 650": "Usually well tolerated at recommended doses. Overdose can cause severe liver damage; allergic reactions are possible.",
    "crocin": "Depends on the exact product. Paracetamol products can cause liver damage in overdose; check the package ingredients.",
    "calpol": "Depends on the exact product. Paracetamol products can cause liver damage in overdose; check the package ingredients.",
    "ibuprofen": "May cause stomach upset, nausea, or indigestion. It can also cause stomach bleeding or kidney problems in some people.",
    "cetirizine": "May cause sleepiness, tiredness, dry mouth, or dizziness.",
    "aspirin": "May cause stomach irritation or bleeding and can trigger allergic reactions in some people.",
    "ecosprin": "May cause stomach irritation or bleeding. Confirm the exact formulation and use it only as advised by a clinician.",
    "ecosprin av": "Combination product: side effects depend on the exact ingredients. Check the package and ask a pharmacist.",
    "combiflam": "Combination product: may cause stomach upset and other ingredient-specific effects. Check the package and ask a pharmacist.",
    "amoxicillin": "May cause nausea, diarrhoea, or rash. Serious allergic reactions can occur.",
    "augmentin": "May cause diarrhoea, nausea, or rash. Serious allergic reactions can occur.",
    "azithromycin": "May cause diarrhoea, nausea, or abdominal discomfort. Seek urgent care for severe allergic symptoms or an unusual heartbeat.",
    "azithral": "May cause diarrhoea, nausea, or abdominal discomfort. Seek medical advice for severe or unusual symptoms.",
    "metformin": "May cause diarrhoea, nausea, gas, or abdominal discomfort. Rarely, a serious condition called lactic acidosis can occur.",
    "omeprazole": "May cause headache, abdominal pain, nausea, diarrhoea, or constipation.",
    "pantoprazole": "May cause headache, diarrhoea, nausea, or abdominal discomfort.",
    "diclofenac": "May cause stomach upset. It can increase the risk of stomach bleeding and cardiovascular or kidney problems in some people."
}


# =========================================================
# 3. LOAD MODEL AND OCR
# =========================================================

@st.cache_resource
def load_model():
    if not MODEL_PATH.exists():
        return None

    try:
        return tf.keras.models.load_model(
            str(MODEL_PATH),
            compile=False
        )
    except Exception as error:
        st.error(f"Could not load the model: {error}")
        return None


@st.cache_resource
def load_ocr():
    return easyocr.Reader(["en"], gpu=False)


# =========================================================
# 4. TEXT NORMALIZATION
# =========================================================

def normalize_text(text):
    """Convert text to lowercase and remove punctuation."""
    text = str(text).lower()
    text = text.replace("\n", " ")
    text = re.sub(r"[^a-z0-9\s]", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# =========================================================
# 5. IMAGE CLASSIFICATION
# =========================================================

def classify_image(image, model):
    """
    IMPORTANT:
    This function assumes the saved model was trained on
    RGB images resized to 260x260 with pixel values 0-255,
    and a single sigmoid output where higher = Real.

    If your training notebook used different preprocessing
    or class mapping, update this function accordingly.
    """

    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))

    image_array = np.array(image, dtype=np.float32)
    image_array = np.expand_dims(image_array, axis=0)

    prediction = model.predict(image_array, verbose=0)
    prediction = np.array(prediction)

    if prediction.size != 1:
        return {
            "label": "Unable to classify",
            "confidence": 0.0,
            "score": None
        }

    score = float(prediction.reshape(-1)[0])

    if not np.isfinite(score):
        return {
            "label": "Unable to classify",
            "confidence": 0.0,
            "score": None
        }

    score = max(0.0, min(1.0, score))

    if score >= 0.70:
        label = "Real-like"
        confidence = score * 100
    elif score <= 0.30:
        label = "Fake-like"
        confidence = (1.0 - score) * 100
    else:
        label = "Uncertain"
        confidence = (0.5 + abs(score - 0.5)) * 100

    return {
        "label": label,
        "confidence": confidence,
        "score": score
    }


# =========================================================
# 6. AUTOMATIC TEXT EXTRACTION USING OCR
# =========================================================

def extract_text(image, reader):
    """
    Read package text from the original image and enhanced
    versions to improve the chance of recognizing text.
    """

    original = np.array(image.convert("RGB"))

    gray = cv2.cvtColor(original, cv2.COLOR_RGB2GRAY)

    clahe = cv2.createCLAHE(
        clipLimit=2.5,
        tileGridSize=(8, 8)
    )

    enhanced = clahe.apply(gray)

    enlarged = cv2.resize(
        enhanced,
        None,
        fx=2,
        fy=2,
        interpolation=cv2.INTER_CUBIC
    )

    blurred = cv2.GaussianBlur(
        enlarged,
        (0, 0),
        2
    )

    sharpened = cv2.addWeighted(
        enlarged,
        1.5,
        blurred,
        -0.5,
        0
    )

    versions = [
        original,
        enlarged,
        sharpened
    ]

    all_lines = {}

    for version in versions:
        try:
            results = reader.readtext(
                version,
                detail=1,
                paragraph=False,
                contrast_ths=0.05,
                adjust_contrast=0.7,
                mag_ratio=1.5
            )

            for item in results:
                detected_text = item[1].strip()
                confidence = float(item[2])

                if not detected_text:
                    continue

                key = normalize_text(detected_text)

                if not key:
                    continue

                if (
                    key not in all_lines
                    or confidence > all_lines[key]["confidence"]
                ):
                    all_lines[key] = {
                        "text": detected_text,
                        "confidence": confidence
                    }

        except Exception:
            continue

    detected_lines = list(all_lines.values())

    combined_text = "\n".join(
        item["text"] for item in detected_lines
    )

    return combined_text, detected_lines


# =========================================================
# 7. AUTOMATIC MEDICINE NAME DETECTION
# =========================================================

def suggest_medicine_name(ocr_text):
    """
    Detect medicine names by matching OCR text against known
    medicine names and common package-label variations.
    """

    normalized = normalize_text(ocr_text)
    compact_text = normalized.replace(" ", "")

    if not normalized:
        return ""

    # Match known names, checking longer names first.
    medicine_names = sorted(
        MEDICINE_INFO.keys(),
        key=lambda name: len(normalize_text(name)),
        reverse=True
    )

    for medicine in medicine_names:
        candidate = normalize_text(medicine)

        if candidate and candidate in normalized:
            return medicine.title()

    # Match common names even if OCR misses spaces.
    aliases = {
        "dolo650": "Dolo 650",
        "paracetamol": "Paracetamol",
        "acetaminophen": "Acetaminophen",
        "combiflam": "Combiflam",
        "augmentin": "Augmentin",
        "azithral": "Azithral",
        "azithromycin": "Azithromycin",
        "amoxicillin": "Amoxicillin",
        "ibuprofen": "Ibuprofen",
        "cetirizine": "Cetirizine",
        "metformin": "Metformin",
        "omeprazole": "Omeprazole",
        "pantoprazole": "Pantoprazole",
        "diclofenac": "Diclofenac",
        "ecosprinav": "Ecosprin AV",
        "ecosprin": "Ecosprin",
        "aspirin": "Aspirin",
        "crocin": "Crocin",
        "calpol": "Calpol"
    }

    for alias, medicine in aliases.items():
        if alias in compact_text:
            return medicine

    # If a known medicine name is not found, try extracting
    # a likely medicine-name line from the OCR output.
    lines = [
        line.strip()
        for line in ocr_text.splitlines()
        if line.strip()
    ]

    for line in lines:
        clean_line = re.sub(
            r"[^A-Za-z0-9\s\-]",
            "",
            line
        ).strip()

        words = clean_line.split()

        # Avoid returning tiny fragments or long instructions.
        if (
            1 <= len(words) <= 3
            and 4 <= len(clean_line) <= 30
            and any(char.isalpha() for char in clean_line)
        ):
            lower_line = clean_line.lower()

            unwanted = [
                "tablet", "tablets", "capsule", "capsules",
                "manufactured", "manufacturing", "expiry",
                "batch", "dosage", "warning", "store",
                "keep out", "composition", "marketed"
            ]

            if not any(term in lower_line for term in unwanted):
                # This is an unverified OCR candidate, not a
                # confirmed medicine identity.
                return clean_line.title()

    return ""


# =========================================================
# 8. EXTRACT PACKAGE DETAILS
# =========================================================

def extract_details(ocr_text):
    text = ocr_text.upper()

    batch_number = "Not detected"
    manufacturing_date = "Not detected"
    expiry_date = "Not detected"

    batch_patterns = [
        r"(?:BATCH|BATCH NO|B\.NO|LOT NO|LOT)\s*[:#\-]?\s*([A-Z0-9\/\-]+)",
        r"\bMFG\.?\s*BATCH\s*[:#\-]?\s*([A-Z0-9\/\-]+)"
    ]

    for pattern in batch_patterns:
        match = re.search(pattern, text)

        if match:
            batch_number = match.group(1).strip(" :-")
            break

    date_pattern = (
        r"\b(0?[1-9]|1[0-2])"
        r"[\/\-\.]"
        r"(20\d{2}|\d{2})\b"
    )

    for match in re.finditer(date_pattern, text):
        date_value = match.group(0)
        start = max(0, match.start() - 30)
        end = min(len(text), match.end() + 15)
        context = text[start:end]

        if any(word in context for word in ["MFG", "MFD", "MANUFACT"]):
            manufacturing_date = date_value

        elif any(word in context for word in ["EXP", "EXPI", "USE BEFORE"]):
            expiry_date = date_value

    return {
        "Batch Number": batch_number,
        "Manufacturing Date": manufacturing_date,
        "Expiry Date": expiry_date
    }


# =========================================================
# 9. MEDICINE INFORMATION LOOKUP
# =========================================================

def lookup_local_medicine(medicine_name):
    normalized_name = normalize_text(medicine_name)

    if not normalized_name:
        return None

    if normalized_name in MEDICINE_INFO:
        info = dict(MEDICINE_INFO[normalized_name])
        info["side_effects"] = COMMON_SIDE_EFFECTS.get(
            normalized_name,
            "Side-effect information is not available in the local database. Check the official leaflet or ask a pharmacist."
        )
        return info

    # Match a detected brand to its dictionary entry.
    for key, info in MEDICINE_INFO.items():
        if key in normalized_name or normalized_name in key:
            matched = dict(info)
            matched["side_effects"] = COMMON_SIDE_EFFECTS.get(
                key,
                "Side-effect information is not available in the local database. Check the official leaflet or ask a pharmacist."
            )
            return matched

    return None


def lookup_medicine_online(medicine_name):
    """
    Optional lookup using the public openFDA drug-label API.
    Online results may be incomplete and are not proof of
    authenticity or approval of the specific package.
    """

    if not medicine_name.strip():
        return None

    url = "https://api.fda.gov/drug/label.json"

    params = {
        "search": (
            f'openfda.brand_name:"{medicine_name}" '
            f'OR openfda.generic_name:"{medicine_name}"'
        ),
        "limit": 1
    }

    try:
        response = requests.get(
            url,
            params=params,
            timeout=8
        )

        if response.status_code != 200:
            return None

        results = response.json().get("results", [])

        if not results:
            return None

        record = results[0]
        openfda = record.get("openfda", {})

        ingredients = openfda.get(
            "substance_name",
            []
        )

        uses = (
            record.get("indications_and_usage")
            or record.get("purpose")
            or []
        )
        side_effects = (
            record.get("adverse_reactions")
            or record.get("warnings")
            or []
        )

        # Keep the main screen readable. Preserve the full label text so the
        # interface can optionally show it in an expander.
        full_uses = uses[0] if uses else "No use information found in the returned label."
        full_side_effects = side_effects[0] if side_effects else (
            "No side-effect information found in the returned label. "
            "Check the package leaflet or ask a pharmacist."
        )

        def short_summary(value, max_chars=520):
            value = re.sub(r"\s+", " ", str(value)).strip()
            # Remove common regulatory section headings where possible.
            value = re.sub(r"^\d+(?:\.\d+)*\s+[A-Z][A-Z /()'-]{4,80}\s+", "", value)
            if len(value) <= max_chars:
                return value
            clipped = value[:max_chars].rsplit(" ", 1)[0]
            return clipped.rstrip(" ,;:-") + "…"

        return {
            "generic": ", ".join(ingredients) if ingredients else "Not listed",
            "use": short_summary(full_uses, 520),
            "side_effects": short_summary(full_side_effects, 520),
            "full_use": full_uses,
            "full_side_effects": full_side_effects
        }

    except (requests.RequestException, ValueError, KeyError):
        return None


# =========================================================
# 10. PAGE DESIGN
# =========================================================

st.title("💊 AI-Based Counterfeit Medicine Detection")
st.write(
    "Upload a medicine package image to extract visible text, "
    "automatically identify a possible medicine name, and run "
    "the trained image-classification model."
)

st.info(
    "Academic demonstration only. The model and OCR cannot "
    "confirm whether a medicine is genuine or safe to consume."
)

with st.sidebar:
    st.header("About the Project")
    st.write("**Model:** EfficientNetB2")
    st.write("**OCR:** EasyOCR")
    st.write("**Image size:** 260 × 260")
    st.write("**Dataset:** Real and Fake package images")
    st.divider()
    st.caption(
        "Name recognition depends on readable package text "
        "and the medicine-name matching rules."
    )


# =========================================================
# 11. UPLOAD IMAGE
# =========================================================

camera_photo = st.camera_input("Take a picture of the medicine package")
uploaded_file = st.file_uploader(
    "Or upload a medicine package image",
    type=["jpg", "jpeg", "png", "webp"]
)

# Prefer the camera photo when one is captured; otherwise use the uploaded file.
image_file = camera_photo if camera_photo is not None else uploaded_file

if image_file is not None:

    try:
        image = Image.open(image_file).convert("RGB")
    except Exception:
        st.error("Could not open this image. Please upload another image.")
        st.stop()

    image_hash = hashlib.md5(
        image_file.getvalue()
    ).hexdigest()[:12]

    left, right = st.columns([1, 1])

    with left:
        st.subheader("Uploaded Image")
        st.image(
            image,
            caption="Medicine package",
            use_container_width=True
        )

    # Load OCR and run it on the uploaded image.
    with st.spinner("Reading package text..."):
        try:
            reader = load_ocr()
            ocr_text, detected_lines = extract_text(image, reader)
            suggested_name = suggest_medicine_name(ocr_text)
            details = extract_details(ocr_text)

        except Exception as error:
            st.error(f"OCR failed: {error}")
            ocr_text = ""
            detected_lines = []
            suggested_name = ""
            details = {
                "Batch Number": "Not detected",
                "Manufacturing Date": "Not detected",
                "Expiry Date": "Not detected"
            }

    with right:
        st.subheader("Automatic Medicine Name Detection")

        if suggested_name:
            st.success(f"Detected medicine name: **{suggested_name}**")
        else:
            st.warning(
                "Medicine name could not be recognized. "
                "Try a clearer, front-facing image showing the name."
            )

        # A new key for each uploaded image prevents a previous
        # medicine name from being retained when a new image is uploaded.
        medicine_name = st.text_input(
            "Correct the name if needed",
            value=suggested_name,
            key=f"medicine_name_{image_hash}",
            placeholder="Enter the medicine name"
        )

        st.caption(
            "The detected name is an OCR-based suggestion. "
            "Check it against the actual package."
        )

    # -----------------------------------------------------
    # CLASSIFICATION
    # -----------------------------------------------------

    st.divider()
    st.subheader("Image Classification")

    model = load_model()

    if model is None:
        st.error(
            f"Model file not found or could not be loaded: {MODEL_PATH}"
        )
        st.write(
            "Place `efficientnetb2_final.keras` in the same folder as `app.py`."
        )
    else:
        with st.spinner("Analyzing image..."):
            result = classify_image(image, model)

        if result["label"] == "Real-like":
            st.success(
                f"Model result: {result['label']} "
                f"({result['confidence']:.1f}% model score)"
            )
        elif result["label"] == "Fake-like":
            st.error(
                f"Model result: {result['label']} "
                f"({result['confidence']:.1f}% model score)"
            )
        else:
            st.warning(
                f"Model result: {result['label']}"
            )

        st.caption(
            "This is a model prediction, not a verified authenticity "
            "result. Confirm that preprocessing and class mapping match "
            "your training notebook."
        )

    # -----------------------------------------------------
    # OCR OUTPUT
    # -----------------------------------------------------

    st.divider()
    st.subheader("Extracted Package Text")

    if ocr_text:
        st.text_area(
            "OCR result",
            value=ocr_text,
            height=180,
            disabled=True,
            key=f"ocr_output_{image_hash}"
        )
    else:
        st.warning("No readable text was found in this image.")

    with st.expander("View OCR confidence scores"):
        if detected_lines:
            for item in detected_lines:
                st.write(
                    f"**{item['text']}** — "
                    f"{item['confidence'] * 100:.1f}% OCR confidence"
                )
        else:
            st.write("No text lines detected.")

    # -----------------------------------------------------
    # PACKAGE DETAILS
    # -----------------------------------------------------

    st.divider()
    st.subheader("Package Details")

    col1, col2, col3 = st.columns(3)

    col1.metric(
        "Batch Number",
        details["Batch Number"]
    )

    col2.metric(
        "Manufacturing Date",
        details["Manufacturing Date"]
    )

    col3.metric(
        "Expiry Date",
        details["Expiry Date"]
    )

    st.caption(
        "Dates and batch numbers are detected only when the label "
        "format is recognized. Verify all details directly on the package."
    )

    # -----------------------------------------------------
    # MEDICINE INFORMATION (AUTOMATIC)
    # -----------------------------------------------------

    st.divider()
    st.subheader("Medicine Information")

    if medicine_name.strip():
        medicine_info = lookup_local_medicine(medicine_name)

        if medicine_info is None:
            with st.spinner("Looking for medicine-label information online..."):
                medicine_info = lookup_medicine_online(medicine_name)

        if medicine_info:
            st.write(f"**Medicine name:** {medicine_name.strip().title()}")
            st.write(f"**Generic / ingredients:** {medicine_info.get('generic', 'Not available')}")
            st.write(f"**Common uses:** {medicine_info.get('use', 'Not available')}")
            st.write(f"**Side effects / warnings:** {medicine_info.get('side_effects', 'Not available. Check the official package leaflet or ask a pharmacist.')}")
            if medicine_info.get("full_use") or medicine_info.get("full_side_effects"):
                with st.expander("View detailed information from the official drug label"):
                    if medicine_info.get("full_use"):
                        st.markdown("**Detailed uses / indications**")
                        st.write(medicine_info["full_use"])
                    if medicine_info.get("full_side_effects"):
                        st.markdown("**Detailed side effects / warnings**")
                        st.write(medicine_info["full_side_effects"])
            st.caption(
                "Information is a general summary, not personal medical advice. "
                "Online label results may be for a different country or formulation; verify the active ingredients and strength on your package."
            )
        else:
            st.warning(
                "The medicine name was not found in the available information sources. "
                "Check the extracted name, enter the exact name and strength if known, "
                "or ask a pharmacist. The app will not guess uses or side effects."
            )
    else:
        st.info("The medicine name has not been identified yet. Upload a clearer, close-up photo showing the printed name and strength.")

    st.warning(
        "Do not take or stop a medicine based on this app. A photo-based model cannot prove authenticity. "
        "Verify the batch and product with the manufacturer or a licensed pharmacist."
    )

    # -----------------------------------------------------
    # FOOTER
    # -----------------------------------------------------

    st.divider()
    st.caption(
        "Important: This application is a student project. "
        "A package image alone cannot establish medicine authenticity. "
        "For safety, verify the product, batch number and expiry date "
        "with a licensed pharmacist or the manufacturer."
    )

else:
    st.info(
        "Take a picture or upload a JPG, JPEG, PNG or WEBP image to begin automatic detection."
    )
