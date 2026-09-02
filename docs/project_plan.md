# Clinic Chatbot Project Plan

## Overview
Build a simple chatbot for one clinic to handle fixed appointments and walk-in visits using an ML model to predict visit busyness.

## Project Structure
- `docs/`: Project documentation and tracking.
- `backend/`: FastAPI backend with an NLP State Machine and Scikit-Learn ML Model.
- `frontend/`: React + Vite frontend for a sleek chat interface.

## Tasks

### Phase 1: Planning and Setup (Current)
- [x] Create project plan in `docs/project_plan.md`
- [x] Define backend and frontend architecture
- [ ] Initialize frontend and backend repositories/folders

### Phase 2: Backend & ML Model
- [ ] Generate synthetic data based on sample visits
- [ ] Train ML model for busyness prediction (Busy/Normal/Free)
- [x] Setup FastAPI server and deterministic NLP State Machine
- [x] Wire ML model into Walk-in State Flow
- [ ] Implement API endpoints for chat interface

### Phase 3: Frontend
- [ ] Initialize React + Vite project
- [ ] Build chat interface (premium styling, clean typography)
- [ ] Connect frontend to backend API for real-time interaction

### Phase 4: Integration & Testing
- [ ] End-to-end testing of appointment booking flow
- [ ] End-to-end testing of walk-in flow and ML predictions
- [ ] Final UI/UX polish
