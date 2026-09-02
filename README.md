# Clinic Chatbot - Appointment & Visit-Time Suggestion

## 🏗️ Architecture Overview
The project is divided into a robust backend and a modern frontend:

* **Backend:** Built with Python and FastAPI. It exposes the chat API endpoint and utilizes the **Google Gemini LLM** via the Generative AI SDK to manage the conversation state naturally. It also houses the Machine Learning pipeline.
* **Frontend:** A React + Vite single-page application. It provides a sleek, responsive chat interface with SSE (Server-Sent Events) streaming.
* **Machine Learning:** A Random Forest model built with Scikit-Learn to predict clinic busyness based on time, day, and session.

## 🤖 How the LLM and ML Model Interact
1. When a patient requests a "walk-in" visit or a "fixed appointment", the Gemini LLM gathers the required details in a natural, conversational way.
2. The LLM is equipped with **Function Calling (Tools)** for two primary actions: `predict_busyness` and `book_fixed_appointment`.
3. If a walk-in is requested, the LLM calls `predict_busyness` with the parsed date and hour. The Python backend evaluates the ML model and returns the prediction (Busy/Normal/Free) to the LLM.
4. The LLM reads the ML prediction and formulates the final response (e.g. suggesting quieter times if it's "Busy").


## 🚀 How to Run It

### Prerequisites
* Python 3
* Node.js and npm

### 1. Backend Setup
Open a terminal, navigate to the backend directory, and install the dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

*(Optional)* If you wish to regenerate the synthetic data and retrain the ML model:
```bash
python ml_pipeline.py
```

Start the FastAPI backend server:
```bash
uvicorn main:app --reload
```
*The backend will run on `http://localhost:8000`.*

### 2. Frontend Setup
Open a **new** terminal window, navigate to the frontend directory, and start the React app:

```bash
cd frontend
npm install
npm run dev
```

*The chat interface will now be available in your browser at `http://localhost:5173`.*
