# 🛡️ Real-time Weapon Detection System

A high-performance AI-powered system designed to detect weapons (Pistols, Knives) in real-time from video streams or webcams. Built with **YOLOv7**, **Flask**, and **OpenCV**.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue)
![YOLOv7](https://img.shields.io/badge/Model-YOLOv7-green)
![Flask](https://img.shields.io/badge/Web-Flask-red)

## 🌟 Features

- **Real-time Detection:** Instantly identifies weapons in video feeds with high accuracy.
- **Web Interface:** View the live detection stream directly in your browser.
- **Immediate Alerts:** Visual bounding boxes around detected threats.
- **User Controls:** Graceful server shutdown via the web UI.
- **Optimized Performance:** Uses a custom `lib` structure for cleaner deployment.

## 🚀 Installation

Follow these steps to set up the project on your local machine.

### Prerequisites
- Python 3.8 or higher installed.

### 1. Clone the Repository
```bash
git clone https://github.com/spacealab/Weapon-Detection-App.git
cd Weapon-Detection-App
```

### 2. Create a Virtual Environment (Optional but Recommended)
```bash
python -m venv .venv
# Activate on Linux/Mac:
source .venv/bin/activate
# Activate on Windows:
.venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

## 🎮 Usage

1. **Run the Application:**
   ```bash
   python detect.py
   ```
   
2. **Access the Dashboard:**
   - The system will automatically open your default browser.
   - If not, navigate to: `http://127.0.0.1:8000`

3. **Stop the System:**
   - Click the **"STOP SERVER"** button on the webpage.
   - Or press `Ctrl+C` in the terminal.

## 🛠️ Technology Stack

- **Core:** Python
- **AI Model:** YOLOv7 (Custom Trained)
- **Computer Vision:** OpenCV
- **Web Server:** Flask

## 📄 License

This project is open-source and available for educational and research purposes.

---
*Developed by Ali (2026)*
