import os
import json
import logging
import asyncio
import pickle
import pandas as pd
from datetime import datetime
import google.generativeai as genai
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)
else:
    print("WARNING: GEMINI_API_KEY not found in environment!")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("clinic_chatbot")

# --- ML MODEL SETUP ---
try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        clf = data['model']
        expected_cols = data['columns']
except Exception as e:
    logger.error(f"Error loading ML model: {e}")
    clf = None
    expected_cols = []

MORNING_HOURS = list(range(8, 12))   # 8, 9, 10, 11
EVENING_HOURS = list(range(16, 22))  # 16, 17, 18, 19, 20, 21
VALID_HOURS = MORNING_HOURS + EVENING_HOURS

def format_hour(hour: int) -> tuple:
    am_pm = "AM" if hour < 12 else "PM"
    disp = hour if hour <= 12 else hour - 12
    disp = 12 if disp == 0 else disp
    return disp, am_pm

def get_booking_count(date_str: str, time_str: str) -> int:
    try:
        csv_path = "data/sample_visits.csv"
        if not os.path.exists(csv_path):
            return 0
        df = pd.read_csv(csv_path)
        if 'Date' not in df.columns or 'Visit Time' not in df.columns:
            return 0
        return len(df[(df['Date'] == date_str) & (df['Visit Time'] == time_str)])
    except Exception:
        return 0

# --- GEMINI TOOLS ---
def predict_busyness(date_str: str, hour: int) -> str:
    """
    Predicts if the clinic is Busy, Normal, or Free for a walk-in visit.
    Use this when a user wants to walk in.
    Clinic valid hours: Morning [8, 9, 10, 11], Evening [16, 17, 18, 19, 20, 21].
    Args:
        date_str: Date in YYYY-MM-DD format.
        hour: The hour in 24-hour format (e.g., 9 for 9 AM, 17 for 5 PM).
    """
    if hour not in VALID_HOURS:
        disp_h, am_pm = format_hour(hour)
        return f"Closed. {disp_h}:00 {am_pm} is outside working hours (8AM-12PM and 4PM-10PM)."

    if not clf:
        return "Normal"

    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day = dt.strftime("%a")
        session = "Morning" if hour < 12 else "Evening"
        
        features = {'Hour': hour}
        for d in ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']:
            features[f"Day_{d}"] = 1 if d == day else 0
        for s in ['Morning', 'Evening']:
            features[f"Session_{s}"] = 1 if s == session else 0
            
        df = pd.DataFrame([features])
        for col in expected_cols:
            if col not in df.columns:
                df[col] = 0
        df = df[expected_cols]
        
        prediction = clf.predict(df)[0]
        return prediction
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown"

def book_fixed_appointment(date_str: str, hour: int, name: str, email: str) -> str:
    """
    Attempts to book a guaranteed fixed appointment.
    Use this only after collecting date, hour, name, and email from the user.
    Args:
        date_str: Date in YYYY-MM-DD format.
        hour: Hour in 24-hour format (e.g., 9 for 9 AM, 17 for 5 PM).
        name: Full name of the patient.
        email: Email address of the patient.
    Returns:
        A string indicating success or failure due to capacity limits.
    """
    if hour not in VALID_HOURS:
        disp_h, am_pm = format_hour(hour)
        return f"Failed: {disp_h}:00 {am_pm} is outside working hours (8AM-12PM and 4PM-10PM)."
        
    disp_h, am_pm = format_hour(hour)
    time_str = f"{disp_h:02d}:00 {am_pm}"
    
    # Capacity check (max 2)
    if get_booking_count(date_str, time_str) >= 2:
        return f"Failed: Slot {time_str} is fully booked. Ask the user for a different time."
        
    # Book it
    try:
        dt = datetime.strptime(date_str, "%Y-%m-%d")
        day_str = dt.strftime("%a")
        os.makedirs("data", exist_ok=True)
        csv_path = "data/sample_visits.csv"
        write_header = not os.path.exists(csv_path)
        with open(csv_path, "a") as f:
            if write_header:
                f.write("Direct patients,Date,Day,Visit Time\n")
            f.write(f"{name},{date_str},{day_str},{time_str}\n")
        return "Success: Appointment booked and saved."
    except Exception as e:
        return f"Failed to save booking: {str(e)}"

def get_system_instruction():
    today = datetime.now()
    return f"""
You are a helpful Clinic Assistant chatbot.
The clinic's working hours are:
- Morning: 8:00 AM – 12:00 PM
- Evening: 4:00 PM – 10:00 PM
Today is {today.strftime('%A')}, {today.strftime('%Y-%m-%d')}. Keep this in mind when the user says "tomorrow".

Core Instructions:
1. Greet the patient and ask if they want a **fixed appointment** (guaranteed slot) or a **walk-in visit**.
2. **Fixed Appointment Path:**
   - Collect their preferred date & time, full name, and email.
   - Use the `book_fixed_appointment` tool.
   - If it succeeds, confirm the booking and tell them a confirmation email was sent.
   - If it fails (fully booked), suggest a nearby time.
3. **Walk-in Path:**
   - Ask what time they plan to come.
   - Use the `predict_busyness` tool.
   - If the prediction is "Busy", tell them it's expected to be busy and suggest 1-2 nearby quieter times (e.g. +/- 1 hour, making sure they are within working hours). You can call the tool again with adjacent hours to check if they are busy before suggesting them.
   - If the prediction is "Normal" or "Free", confirm it's a good time to visit.
4. Keep your answers concise, friendly, and properly formatted in markdown.
"""

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class MessageReq(BaseModel):
    session_id: str
    message: str

chat_sessions = {}

def get_chat_session(session_id: str):
    if session_id not in chat_sessions:
        if not api_key:
            raise Exception("GEMINI_API_KEY is not set.")
        model = genai.GenerativeModel(
            model_name='gemini-1.5-flash',
            system_instruction=get_system_instruction(),
            tools=[predict_busyness, book_fixed_appointment]
        )
        chat_sessions[session_id] = model.start_chat(enable_automatic_function_calling=True)
    return chat_sessions[session_id]

@app.post("/api/chat")
async def chat_endpoint(req: MessageReq):
    try:
        session = get_chat_session(req.session_id)
        
        # We need a generator to yield streaming chunks compatible with SSE
        async def generate():
            response = session.send_message(req.message, stream=True)
            for chunk in response:
                if chunk.text:
                    data = {"content": chunk.text}
                    yield f"data: {json.dumps(data)}\n\n"
                    
        return StreamingResponse(generate(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        async def generate_error():
            msg = f"Sorry, I encountered an error: {str(e)}\n\nMake sure GEMINI_API_KEY is set in your environment or .env file."
            yield f"data: {json.dumps({'content': msg})}\n\n"
        return StreamingResponse(generate_error(), media_type="text/event-stream")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
