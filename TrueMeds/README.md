# 💊 AI-Based Counterfeit Medicine Detection System

An AI-powered medicine package analysis application that combines **deep learning, image processing, Optical Character Recognition (OCR), and medicine information retrieval** to analyze medicine packaging.

The application uses a trained **EfficientNetB2** model to classify medicine images as Fake or Real, **EasyOCR** to extract text from medicine packaging, and **Streamlit** to provide an interactive user interface.

It also attempts to identify the medicine name, retrieve available information about its uses and side effects, and extract the batch number, manufacturing date, and expiry date when the text is readable.

> **Disclaimer:** This project is an academic demonstration. Image-based classification cannot establish whether a medicine is genuine, counterfeit, or safe to consume. Always verify medicines with a licensed pharmacist, manufacturer, or appropriate regulatory authority.

---

## 📌 Table of Contents

- [Project Overview](#-project-overview)
- [Objectives](#-objectives)
- [Features](#-features)
- [System Architecture](#-system-architecture)
- [Technologies Used](#-technologies-used)
- [Deep Learning Model](#-deep-learning-model)
- [Dataset Description](#-dataset-description)
- [How the Application Works](#-how-the-application-works)
- [Project Structure](#-project-structure)
- [Requirements](#-requirements)
- [Installation](#-installation)
- [Running the Application](#-running-the-application)
- [Medicine Information Retrieval](#-medicine-information-retrieval)
- [Example Output](#-example-output)
- [Limitations](#-limitations)
- [Future Scope](#-future-scope)
- [Applications](#-applications)
- [Conclusion](#-conclusion)
- [Author](#-author)

---

## 📖 Project Overview

Counterfeit medicines are a public health concern because they may contain incorrect ingredients, insufficient active ingredients, harmful substances, or misleading packaging information.

This project demonstrates how artificial intelligence and image processing can be combined to analyze medicine package images and retrieve related medicine information.

The user can upload a medicine photograph or capture one using the application's camera interface. The system processes the image, generates a classification using a previously trained deep-learning model, extracts visible text using OCR, and attempts to identify the medicine.

When a medicine name is identified, the application searches its local medicine-information database and may retrieve additional drug-label information from the OpenFDA API.

The application displays the available information through an interactive Streamlit interface.

---

## 🎯 Objectives

The main objectives of this project are:

1. To develop an AI-based application for medicine package image classification.
2. To use EfficientNetB2 for binary image classification.
3. To extract text from medicine packaging using EasyOCR.
4. To identify possible medicine names from the extracted text.
5. To display available information about medicine uses and side effects.
6. To extract batch numbers, manufacturing dates, and expiry dates when readable.
7. To integrate image classification and OCR into a single application.
8. To provide an easy-to-use interface using Streamlit.
9. To demonstrate the practical application of deep learning and computer vision.

---

## ✨ Features

### 1. Medicine Image Classification

- Uses a trained EfficientNetB2 model.
- Accepts medicine package images as input.
- Produces a Fake or Real classification according to the trained model.
- Displays the model's prediction score when available.

**Important:** The prediction is an image-model result, not verified evidence of authenticity.

### 2. Medicine Name Detection

- Uses EasyOCR to recognize visible text.
- Extracts possible medicine names from the package.
- Matches recognized text against a local medicine-information database.
- Attempts an online lookup when a suitable local match is unavailable.

### 3. Medicine Information

Displays available information about:

- Medicine name
- Generic name or active ingredients
- Common uses
- Side effects
- Relevant warnings

Information availability depends on the recognized medicine name and the data returned by the configured information source.

### 4. Batch Number Extraction

Attempts to identify batch or lot numbers using OCR and text-pattern matching.

### 5. Manufacturing Date Extraction

Attempts to recognize manufacturing dates from visible package labels such as `MFG` or `MFD`.

### 6. Expiry Date Extraction

Attempts to recognize expiry dates from visible package labels such as `EXP` or `USE BEFORE`.

Date extraction depends on image quality, OCR accuracy, and the date formats supported by the application.

### 7. Image Enhancement

Image processing may include:

- Grayscale conversion
- Contrast enhancement
- Image resizing
- Sharpening

These operations may improve OCR performance on some images but cannot recover information that is not visible.

### 8. Interactive Web Interface

- Built using Streamlit.
- Supports image upload.
- Supports camera capture when enabled in the application.
- Displays classification and medicine information in one interface.
- Provides an expandable section for detailed drug-label information when available.

---

## 🏗️ System Architecture

The application follows the workflow below:

```text
                 USER
                   |
                   v
       Capture or Upload Image
                   |
                   v
           Image Preprocessing
                   |
          +--------+---------+
          |                  |
          v                  v
   EfficientNetB2          EasyOCR
   Classification       Text Extraction
          |                  |
          v                  v
    Fake / Real          Recognized Text
                             |
                             v
                    Medicine Name Matching
                             |
                             v
                  +----------+----------+
                  |                     |
                  v                     v
             Local Database        OpenFDA API
                  |                     |
                  +----------+----------+
                             |
                             v
                  Medicine Information
                  - Generic Name
                  - Uses
                  - Side Effects
                             |
                             v
                   Batch / Date Extraction
                             |
                             v
                    Streamlit Results
```

The classification and OCR components perform different tasks. The classification model predicts an image category, while OCR extracts visible text.

---

## 🛠️ Technologies Used

| Technology | Purpose |
|---|---|
| Python | Main programming language |
| TensorFlow | Deep-learning framework |
| Keras | Model loading and prediction |
| EfficientNetB2 | Medicine image classification |
| EasyOCR | Text recognition from images |
| OpenCV | Image preprocessing and enhancement |
| NumPy | Numerical operations and image arrays |
| Pillow (PIL) | Image loading and conversion |
| Streamlit | Interactive web application |
| Requests | HTTP requests to an online API |
| OpenFDA API | Retrieval of available drug-label information |
| Regular Expressions | Extraction of medicine names, batch numbers, and dates |
| pathlib | File path management |
| hashlib | Generation of image-content hashes |

---

## 🧠 Deep Learning Model

### EfficientNetB2

EfficientNetB2 is a convolutional neural network architecture designed for image classification.

It uses a balanced scaling approach to adjust network depth, width, and image resolution.

In this project, the trained model is used to classify medicine package images into two categories:

- **Fake**
- **Real**

### Model Input

The model was configured for images of size:

```text
260 × 260 pixels
```

The image preprocessing used during prediction must match the preprocessing used during model training.

### Model File

The application uses the trained Keras model:

```text
efficientnetb2_final.keras
```

The model must be available at the path expected by the application.

### Model Evaluation

The model should be evaluated using the held-out test dataset.

Recommended evaluation metrics include:

- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix

Actual test results should be reported only after evaluating the trained model. No accuracy value is claimed in this README because a verified test result has not been provided here.

---

## 📂 Dataset Description

The project uses a labelled dataset containing medicine package images in two categories.

The dataset split provided for this project is:

| Dataset Split | Fake Images | Real Images | Total Images |
|---|---:|---:|---:|
| Training | 284 | 284 | 568 |
| Validation | 36 | 61 | 97 |
| Testing | 36 | 61 | 97 |
| **Total** | **356** | **406** | **762** |

### Training Set

The training set is used to learn patterns from labelled images.

### Validation Set

The validation set is used to monitor training performance and support model selection.

### Testing Set

The test set is used to evaluate the model on images that were not used for training or validation.

### Dataset Considerations

The model's performance depends on:

- Image quality
- Correct image labels
- Dataset diversity
- Packaging variations
- Similarity between training images and real-world inputs

If the dataset contains unclear images or the model learns irrelevant visual patterns, predictions may be unreliable.

The number of images alone does not establish that the model can reliably identify counterfeit medicines.

---

## ⚙️ How the Application Works

### Step 1: Capture or Upload an Image

The user captures a medicine photograph or uploads an existing image.

Supported formats may include:

- JPG
- JPEG
- PNG
- WEBP

### Step 2: Image Preprocessing

The image is loaded and prepared for the model. Image enhancement may also be applied to help OCR recognize text.

### Step 3: Image Classification

The preprocessed image is passed to the trained EfficientNetB2 model.

The model generates a classification score that the application maps to a category according to the model's output configuration.

The class mapping must match the labels used during training.

### Step 4: OCR Text Extraction

EasyOCR analyzes the package image and attempts to recognize printed text.

Recognized text may include:

- Medicine name
- Strength
- Batch number
- Manufacturing date
- Expiry date

Not every field will be detected in every image.

### Step 5: Medicine Name Identification

The recognized text is normalized and compared with medicine names in the local database.

If a suitable match is found, the corresponding medicine information is retrieved.

### Step 6: Online Medicine Information Lookup

If the medicine is not found locally, the application may query the OpenFDA drug-label API.

The returned information depends on the API response and the accuracy of the identified medicine name.

### Step 7: Display Results

The Streamlit interface displays the available classification, medicine information, and extracted package details.

Fields that cannot be recognized should be shown as `Not detected` rather than being guessed.

---

## 📁 Project Structure

The exact structure depends on the files present in your project directory. A possible organization is:

```text
ML Project/
│
├── app.py
├── efficientnetb2_final.keras
├── efficientnetb2_best.keras
│
├── final_dataset/
│   ├── train/
│   │   ├── Fake/
│   │   └── Real/
│   │
│   ├── val/
│   │   ├── Fake/
│   │   └── Real/
│   │
│   └── test/
│       ├── Fake/
│       └── Real/
│
├── medicine_database.csv
│
├── requirements.txt
│
└── README.md
```

**Note:** This is an illustrative structure. Keep only the files and folders that actually exist in your project. The application and trained model paths must match your configuration.

---

## 📋 Requirements

Install the Python packages used by the application.

Typical dependencies include:

```text
tensorflow
streamlit
easyocr
opencv-python-headless
numpy
Pillow
requests
```

Additional packages may be required depending on the exact code and environment.

Use a Python version compatible with the installed TensorFlow and EasyOCR versions.

---

## 💻 Installation

### Step 1: Install Python

Install a compatible Python version if it is not already installed.

Check the installed version:

```bash
python --version
```

### Step 2: Open the Project Directory

Navigate to the folder containing `app.py` and the trained model.

For example, on Windows:

```powershell
cd "C:\ML Project (1)\ML Project"
```

Replace the path if your project is stored elsewhere.

### Step 3: Create a Virtual Environment

```powershell
python -m venv venv
```

Activate it:

```powershell
.\venv\Scripts\Activate.ps1
```

If you use Command Prompt instead of PowerShell, activate it with:

```cmd
venv\Scripts\activate.bat
```

### Step 4: Install Dependencies

Install the required packages:

```powershell
python -m pip install --upgrade pip
python -m pip install tensorflow streamlit easyocr opencv-python-headless numpy Pillow requests
```

If you already have a working environment with compatible versions, you do not need to reinstall every package.

### Step 5: Verify the Model File

Confirm that the trained model exists in the expected location.

```powershell
dir efficientnetb2_final.keras
```

If the model is stored elsewhere, update the path in `app.py` or copy the model into the configured location.

---

## ▶️ Running the Application

From the project directory, run:

```powershell
python -m streamlit run app.py
```

Streamlit will display a local address in the terminal. Open that address in your browser.

### Using the Application

1. Open the Streamlit application.
2. Capture or upload a medicine package image.
3. Wait for image classification and OCR processing.
4. Review the predicted class.
5. Check the extracted medicine name.
6. Review available uses and side-effect information.
7. Check the batch number and date fields.
8. If details are missing, try a clearer photograph.

For better OCR results:

- Use good lighting.
- Keep the package text in focus.
- Avoid reflections.
- Photograph the printed name and dates closely.
- Keep the camera steady.
- Ensure that the relevant text is visible.

---

## 💊 Medicine Information Retrieval

The application combines local medicine information with online drug-label lookup.

### Local Medicine Database

A local dictionary or CSV file can store information about supported medicines, such as:

- Brand name
- Generic name
- Uses
- Side effects

The local lookup works only for medicines represented in the configured database and supported matching rules.

### OpenFDA API

The OpenFDA drug-label API provides access to available drug-label information.

The application may retrieve fields such as:

- Active ingredients
- Indications and usage
- Adverse reactions
- Warnings

Official drug-label sections can be lengthy and may describe serious warnings rather than only common side effects.

The information returned by the API should be checked against the exact medicine, ingredients, strength, and formulation.

**Important:** An API match is not proof that the uploaded package is authentic. It also does not establish that the medicine is appropriate for a particular patient.

---

## 🖥️ Example Output

The following is an illustrative example of the information the application may display.

```text
AI-Based Counterfeit Medicine Detection System

Medicine Classification: Real
Model Score: Example score only

Medicine Name: Paracetamol
Generic Name: Paracetamol

Common Uses:
Relief of pain and reduction of fever.

Side Effects:
Possible adverse effects depend on the product and its use.
Excessive paracetamol can cause severe liver damage.

Batch Number: Not detected
Manufacturing Date: Not detected
Expiry Date: Not detected

Note:
Image classification cannot verify medicine authenticity.
Please confirm medicine details with a pharmacist.
```

The example above is not a real prediction or medical assessment. Actual results depend on the uploaded image, trained model, OCR output, and information source.

---

## ⚠️ Limitations

1. **Image-based classification:** The model learns visual patterns from its training dataset and may misclassify unfamiliar packaging.

2. **Dataset quality:** Blurry, poorly labelled, or insufficiently diverse images can affect performance.

3. **OCR accuracy:** Small fonts, glare, curved packaging, and low resolution can prevent accurate text extraction.

4. **Medicine identification:** Incorrect OCR text may result in an incorrect medicine match.

5. **Online information:** Drug-label information may be missing, incomplete, or inappropriate for the exact formulation.

6. **Date extraction:** The application may not recognize every date format or label arrangement.

7. **No authenticity verification:** The application does not independently verify manufacturer records, regulatory status, or batch authenticity.

8. **Medical safety:** The application cannot determine whether a medicine is safe or suitable for an individual patient.

9. **Internet dependency:** Online medicine information retrieval requires network access.

10. **Performance limitations:** Model accuracy must be established through proper testing and should not be inferred from individual successful predictions.

---

## 🚀 Future Scope

Possible future improvements include:

- Expanding the dataset with more diverse and clearly labelled images.
- Evaluating the model on unseen brands and packaging styles.
- Improving OCR for low-quality images.
- Adding multilingual medicine-name recognition.
- Improving medicine-name matching using fuzzy matching and confidence thresholds.
- Adding manufacturer-authorized batch verification.
- Connecting to suitable official medicine databases.
- Improving the extraction of manufacturing and expiry dates.
- Adding structured reports and downloadable results.
- Developing a mobile-friendly interface.
- Introducing uncertainty handling for low-confidence predictions.
- Performing more extensive evaluation with precision, recall, F1-score, and confusion matrices.

These improvements could make the application more useful, but authenticity verification would still require reliable external evidence.

---

## 🌍 Applications

Potential educational and research applications include:

- Demonstrating AI-based image classification.
- Demonstrating OCR on pharmaceutical packaging.
- Extracting printed information from medicine labels.
- Exploring medicine-information retrieval.
- Studying computer vision and deep learning.
- Building a prototype for future pharmaceutical packaging verification research.

This application should not be used as the sole basis for purchasing, taking, rejecting, or discontinuing a medicine.

---

## 🏁 Conclusion

The **AI-Based Counterfeit Medicine Detection System** demonstrates the integration of deep learning, image processing, OCR, online data retrieval, and web application development.

EfficientNetB2 performs image classification, EasyOCR extracts visible text, and the medicine-information component attempts to retrieve information about identified medicines. Streamlit brings these components together in an interactive interface.

The project provides an educational demonstration of how AI can support medicine-package analysis. However, image classification and text recognition alone cannot prove that a medicine is genuine or safe.

Reliable counterfeit-medicine verification requires appropriate manufacturer, regulatory, or other trusted evidence.

---

## 👩‍💻 Author

**Anjali Vaibhav Palake**

Third-Year B.Tech Information Technology Student

Government College of Engineering, Karad

GitHub: [Anjali-Palake755](https://github.com/Anjali-Palake755)

LinkedIn: [Anjali Palake](https://www.linkedin.com/in/anjali-palake-bb509a33)

---