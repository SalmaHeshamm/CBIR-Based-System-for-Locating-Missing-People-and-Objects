# 🔍 CBIR-Based System for Locating Missing People and Objects

> A deep learning-powered Content-Based Image Retrieval (CBIR) system that helps locate missing persons and objects through intelligent image matching.

---

## 📌 Overview

Traditional methods for finding missing people or objects — posting on social media, printing flyers, or manual searching — are slow, imprecise, and often ineffective. This project tackles that problem by building a **CBIR system** backed by **ResNet deep learning** that lets users upload an image and instantly find visually similar matches in a database.

---

## 🎬 Demo

https://github.com/user-attachments/assets/fbe04ca5-5550-48ec-9881-41856b9d9b2c

---

## ✨ Features

- 📷 **Image-based search** — upload a photo to find matching persons or objects (no text descriptions needed)
- 🧠 **ResNet feature extraction** — uses convolutional layers to capture rich, high-level visual features
- 📐 **Cosine similarity matching** — scale- and illumination-invariant image comparison
- ⚡ **Near real-time retrieval** — average query time of **0.0791 seconds**
- 🌐 **Web application** — accessible, responsive UI built with Django
- 🗄️ **Scalable database** — stores image feature vectors for efficient large-scale retrieval
- 📋 **Missing person registration** — users can report and register missing persons or items

---

## 🚀 Getting Started

### Prerequisites

```bash
Python 3.8+
pip
virtualenv (recommended)
```

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/SalmaHeshamm/CBIR-Based-System-for-Locating-Missing-People-and-Objects.git
cd CBIR-Based-System-for-Locating-Missing-People-and-Objects

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate       # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Apply database migrations
python manage.py migrate

# 5. Run the development server
python manage.py runserver
```

Then open your browser at `http://127.0.0.1:8000`

### Extracting & Indexing Features

```bash
# Pre-extract and store feature vectors for your database images
python extract_features.py --image_dir /path/to/images
```
