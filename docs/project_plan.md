# Clinic Chatbot Project Plan

## Overview
Build a full-stack clinic chatbot with a hybrid architecture designed to handle two main flows efficiently:
- **Fixed Appointments (NLP Flow):** Routine tasks like booking fixed appointments, collecting user details, and basic greetings are handled entirely locally using a deterministic NLP State Machine to bypass API rate limits.
- **Walk-in Visits (ML + LLM Flow):** When a patient requests a walk-in, a local Scikit-Learn ML model predicts clinic busyness based on the time. The backend then structures this data and makes a single call to a Dual LLM (Ollama or Gemini) to generate a polite, natural-sounding response as a clinic receptionist.

## Project Structure
- `docs/`: Project documentation and tracking.
- `backend/`: FastAPI backend with an NLP State Machine, Scikit-Learn ML Model, and Dual LLM Integration.
- `frontend/`: React + Vite frontend for a sleek chat interface.

## Tasks

### Phase 1: Planning and Setup (Completed)
- [x] Create project plan in `docs/project_plan.md`
- [x] Define backend and frontend architecture
- [x] Initialize frontend and backend repositories/folders

### Phase 2: Backend, ML & LLM Integration (Completed)
- [x] Generate synthetic data based on sample visits
- [x] Train ML model for busyness prediction (Busy/Normal/Free)
- [x] Setup FastAPI server and deterministic NLP State Machine
- [x] Wire ML model into Walk-in State Flow
- [x] Integrate Dual LLM (Ollama & Gemini API) for response generation
- [x] Implement prompt engineering for the receptionist persona
- [x] Implement API endpoints for chat interface

### Phase 3: Frontend (Completed)
- [x] Initialize React + Vite project
- [x] Build chat interface (premium styling, clean typography)
- [x] Connect frontend to backend API for real-time interaction

### Phase 4: Integration & Testing (Completed)
- [x] End-to-end testing of appointment booking flow
- [x] End-to-end testing of walk-in flow and ML predictions
- [x] Final UI/UX polish
