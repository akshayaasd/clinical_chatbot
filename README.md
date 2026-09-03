# Clinic Chatbot - Appointment & Walk-In Assistant

A full-stack clinic chatbot built to handle both **fixed appointments** and **walk-in predictions**. The system uses a hybrid architecture (NLP State Machine + Machine Learning + Local/Cloud LLM) to guarantee zero API rate limits while still delivering natural, dynamic responses.

## 🏗️ Architecture Overview

The project is divided into a robust backend and a modern frontend:

* **Backend:** Built with Python 3.9 and FastAPI. Handles the conversational state machine natively to prevent API rate limit issues.
* **Frontend:** A React + Vite single-page application providing a sleek, responsive chat interface with SSE (Server-Sent Events) streaming.
* **Machine Learning:** A Random Forest model built with Scikit-Learn. Trained on synthetic clinic data to predict clinic busyness (Busy/Normal/Free) based on time, day, and session.
* **Dual LLM Integration:** Supports both **Ollama (local, free, no limits)** and **Gemini API (cloud)** for natural language generation.

## 🤖 How the LLM and ML Model Interact

To circumvent strict cloud API rate limits (like Gemini's 5 req/min free tier), this project uses a highly efficient **Hybrid Architecture**:

1. **Native NLP State Machine:** Routine conversation tasks (greetings, asking for patient name/email/date, booking fixed appointments) are handled entirely locally using regular expressions and Python logic. **Zero LLM calls are made here.**
2. **Local Machine Learning:** When a patient asks for a walk-in, the system parses the time and feeds it into the Scikit-Learn `RandomForestClassifier` locally to predict busyness.
3. **Single LLM Call for Polish:** After the ML model predicts the busyness (and checks surrounding hours if busy), the backend packages this raw data into a prompt and makes a *single* call to the LLM. The LLM acts purely as a "receptionist persona" to formulate a polite, clinical response based on the data.
4. **Fallback Options:** You can configure the LLM provider to use `ollama` (default, no rate limits) or `gemini`.

## 🚀 How to Run It

### Prerequisites
* Python 3
* Node.js and npm
* [Ollama](https://ollama.com/) (Optional but recommended for local, free LLM inference)

### 1. Backend Setup
Open a terminal, navigate to the backend directory, and install the dependencies:

```bash
cd backend
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

**Environment Variables:**
Copy the example environment file and adjust if necessary:
```bash
cp .env.example .env
```
By default, `LLM_PROVIDER=auto` will try to use Ollama (`llama3`). If Ollama isn't running and you've provided a `GEMINI_API_KEY`, it will fall back to Gemini automatically.

*(Optional)* If you wish to regenerate the synthetic data and retrain the ML model for the current dates:
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

## 📂 Project Structure

- `backend/main.py`: The core FastAPI server, state machine, and LLM router.
- `backend/ml_pipeline.py`: Script to generate synthetic data based on real patterns and train the Random Forest model.
- `backend/data/sample_visits.csv`: The clinic's actual booked appointment log.
- `frontend/src/App.jsx`: The React chat interface.
