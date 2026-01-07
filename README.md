# VERITAS HEALTH AGENT 🏥🤖

> **Neutralizing health misinformation with AI-powered analysis.**

Veritas Health Agent is a full-stack AI-powered application designed to analyze, verify, and correct health claims. By combining a modern React frontend with a powerful LangGraph-based Python backend, it fights misinformation by providing verified verdicts, confidence scores, and corrective information from trusted medical sources (WHO, CDC, Mayo Clinic, etc.).

---

## ✨ Features

- **🛡️ Fact-Checking**: Instantly determines if a claim is **True**, **False**, or **Misleading**.
- **🔍 Multi-Modal Analysis**: Capable of analyzing both text claims and text extracted from images (OCR).
- **🧠 Advanced AI Agent**: Powered by **Google Gemini 1.5 Pro** and orchestrated via **LangGraph**.
- **🌐 Internet-Connected**: Uses `DuckDuckGo` to cross-reference claims with real-time web data.
- **📚 Resilient Fallback**: Includes a robust "Mock Mode" with 100+ verified health myths for offline reliability.
- **🎨 Glassmorphism UI**: A beautiful, modern interface built with React and Vanilla CSS.
- **🚑 Safety First**: Built-in quality assurance prompts ensure professional, neutral, and safe advice.

---

## 🛠️ Tech Stack

### Backend
- **Framework**: FastAPI
- **AI/ML**: LangChain, LangGraph, Google Gemini 1.5 Pro
- **Tools**: DuckDuckGo Search, Gemini Vision (OCR)
- **Language**: Python 3.x

### Frontend
- **Framework**: React 19 + Vite
- **Styling**: Vanilla CSS (Glassmorphism Design System)
- **Language**: JavaScript

---

## 🚀 Getting Started

### Prerequisites
- Python 3.8+
- Node.js & npm
- A Google Cloud API Key (for Gemini)

### 1. Clone the Repository
```bash
git clone <repository-url>
cd MumbaiHacks2025
```

### 2. Backend Setup
Navigate to the backend folder and set up the environment.

```bash
cd backend

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Mac/Linux:
# source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

**Configuration**:
Create a `.env` file in the `backend/` directory:
```env
GOOGLE_API_KEY=your_google_api_key_here
```

**Run the Server**:
```bash
uvicorn main:app --reload
```
The API will run at `http://localhost:8000`.

### 3. Frontend Setup
Open a new terminal and navigate to the frontend folder.

```bash
cd frontend

# Install dependencies
npm install

# Run the development server
npm run dev
```
The app will run at `http://localhost:5173`.

---

## 📖 Usage

1. Open the frontend in your browser.
2. Enter a health claim (e.g., *"Drinking bleach cures COVID-19"* or *"Turmeric helps with inflammation"*).
3. Click **Verify Claim**.
4. View the **Verdict**, **Confidence Score**, **Explanation**, and **Sources**.

---

## 📡 API Endpoints

### `POST /analyze`
Analyzes a text claim or image for misinformation.

**Request Body**:
```json
{
  "text": "Claim text here...",
  "image_base64": "Optional base64 image string..."
}
```

**Response**:
```json
{
  "verdict": "Misleading",
  "confidence": 0.95,
  "explanation": "Detailed analysis...",
  "sources": ["WHO", "CDC"],
  "corrective_information": "Official correction..."
}
```

---

## 📂 Project Structure

```
MumbaiHacks2025/
├── backend/            # FastAPI Server
│   ├── agent.py        # LangGraph Agent Logic
│   ├── main.py         # API Endpoints
│   └── requirements.txt
├── frontend/           # React Application
│   ├── src/
│   │   ├── App.jsx     # Main UI Logic
│   │   └── App.css     # Glassmorphism Styles
│   └── package.json
└── README.md           # Project Documentation
```

---

## 🛡️ License
This project is open-source and available for educational purposes.
