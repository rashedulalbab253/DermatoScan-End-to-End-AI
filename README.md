# DermatoScan AI: End-to-End Skin Cancer Classification System

[![GitHub License](https://img.shields.io/github/license/rashedulalbab1234/DermatoScan-End-to-End-AI)](LICENSE)
[![Docker Image](https://img.shields.io/badge/docker-ready-blue.svg)](https://hub.docker.com/r/rashedulalbab1234/dermatoscan-ai)
[![FastAPI](https://img.shields.io/badge/FastAPI-005571?style=flat&logo=fastapi)](https://fastapi.tiangolo.com)
[![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org)

**DermatoScan AI** is a professional-grade, full-stack medical imaging application designed for the multi-class classification of skin lesions. Leveraging state-of-the-art Deep Learning (EfficientNet-B3) and a high-performance backend, it provides near-instant diagnostic insights for seven distinct types of skin diseases.

---

## 🚀 Key Features

-   **High-Accuracy Model:** Powered by **EfficientNet-B3**, pre-trained on ImageNet and fine-tuned on the HAM10000 dataset.
-   **Full-Stack Architecture:** Integrated with **FastAPI** for a fast, asynchronous backend and **Jinja2** for a responsive, modern web interface.
-   **Authentication System:** Secure user registration and login system with JWT authentication and salted password hashing (bcrypt).
-   **Prediction Tracking:** Automated database logging of prediction results and confidence levels for every user.
-   **Admin Dashboard:** Comprehensive oversight of system users and diagnostic history.
-   **DevOps Ready:** Fully containerized with Docker and automated CI/CD pipelines via GitHub Actions.

---

## 🛠️ Tech Stack

-   **Deep Learning:** PyTorch, Torchvision (EfficientNet-B3)
-   **Backend:** FastAPI, Uvicorn, Python 3.10
-   **Database:** SQLite (Relational Storage)
-   **Frontend:** HTML5, CSS3, Jinja2 Templates
-   **Security:** JWT (JSON Web Tokens), Bcrypt
-   **Deployment:** Docker, GitHub Actions (CI/CD)

---

## 📦 Project Structure

```text
├── .github/workflows/      # CI/CD pipelines
├── Dataset/                # Image preprocessing and organization scripts
├── static/                 # CSS, images, and user uploads
├── templates/              # HTML views (Jinja2)
├── database.py             # SQLite ORM & Logic
├── main.py                 # FastAPI Application entry point
├── model.py                # Neural Network architecture
├── train.py                # Model training pipeline
├── utils.py                # Data loading & helper functions
└── Dockerfile              # Containerization configuration
```

---

## ⚙️ Installation & Usage

### 1. Local Setup

```bash
# Clone the repository
git clone https://github.com/rashedulalbab1234/DermatoScan-End-to-End-AI.git
cd DermatoScan-End-to-End-AI

# Create and activate virtual environment
python -m venv env
.\env\Scripts\activate  # Windows

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```
Access the app at: `http://localhost:8000`

### 2. Docker Setup

To run the system without any local installation:

```bash
# Pull the latest image
docker pull rashedulalbab1234/dermatoscan-ai:latest

# Run the container
docker run -p 8000:8000 rashedulalbab1234/dermatoscan-ai
```

---

## 🧬 Model Architecture

The core of the system is the **EfficientNet-B3** architecture, chosen for its optimal balance between accuracy and computational efficiency.
-   **Input Size:** 224x224 RGB Images
-   **Optimizer:** Adam with custom Learning Rate Scheduling
-   **Loss Function:** Cross-Entropy Loss
-   **Classification Layers:** Customized dense layers (512 -> 128 -> 7 output classes)

---

## 🤝 Contribution & License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Attribution
Digital foundations for this project were derived from the open-source work of MD. ENAMUL ATIQ and enhanced for a full-stack, end-to-end production environment.

---
**Developed with ❤️ by [Rashed]**