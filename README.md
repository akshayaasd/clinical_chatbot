# Clinic Chatbot - Appointment & Visit-Time Suggestion

## 🏗️ Architecture Overview
The project is divided into a robust backend and a modern frontend:

* **Backend:** Built with Python and FastAPI. It exposes the chat API endpoint and uses a high-speed, deterministic Regex-based NLP state machine to drive the chatbot's reasoning. It also houses the Machine Learning pipeline.
* **Frontend:** A React + Vite single-page application. It provides a sleek, responsive, light-mode chat interface (matching the Credang theme) for users to interact with the bot.
* **Machine Learning:** A Random Forest model built with Scikit-Learn. It is trained on 60 days of synthetic clinic visit data (generated based on the provided sample data) to predict if the clinic will be `Free`, `Normal`, or `Busy`.

## 🤖 How the NLP State Machine and ML Model Interact
1. When a patient requests a "walk-in" visit, the state machine naturally converses to find out what time they plan to arrive.
2. The state machine parses the requested date and time using regex and calls the `predict_busyness` function.
3. The function formats the requested date and time into features (Hour, Day of Week, Morning/Evening Session) and runs an inference using our pre-trained Random Forest model.
4. The ML model returns the predicted busyness state (`Free`, `Normal`, or `Busy`) to the state machine, which formulates the final response.

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
