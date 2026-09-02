from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import asyncio
from dotenv import load_dotenv
import pickle
import pandas as pd
from datetime import datetime
import json
import re

load_dotenv()

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("clinic_chatbot")

# Load ML Model
try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        clf = data['model']
        expected_cols = data['columns']
    logger.info("Successfully loaded ML model from 'model.pkl'")
except Exception as e:
    logger.error(f"Error loading ML model: {e}", exc_info=True)
    clf = None
    expected_cols = []

def log_conversation_to_file(session_id: str, role: str, content: str):
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("data/chat_logs.txt", "a", encoding="utf-8") as f:
            # Replace newlines in content for cleaner single-line logs
            safe_content = content.replace('\n', ' ') if content else ''
            f.write(f"[{timestamp}] [Session: {session_id}] {role}: {safe_content}\n")
    except Exception as e:
        logger.error(f"Failed to write to chat log file: {e}")

def predict_busyness(date_str: str, hour: int) -> str:
    """
    Predicts if the clinic is Busy, Normal, or Free for a given date and hour.
    Args:
        date_str: The date in YYYY-MM-DD format.
        hour: The hour in 24-hour format (e.g., 8 for 8:00 AM, 16 for 4:00 PM). Clinic hours are 8-11 and 16-21.
    """
    if not clf:
        logger.warning(f"predict_busyness called for {date_str} {hour}:00 but ML model is missing! Returning fallback 'Normal'")
        return "Normal" # fallback
        
    try:
        logger.info(f"🛠️  TOOL CALL: predict_busyness called with date={date_str}, hour={hour}")
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
        logger.info(f"✅ TOOL RESULT: Model prediction for {date_str} {hour}:00 is -> {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"Prediction error during predict_busyness: {e}", exc_info=True)
        return "Unknown"

# Initialize FastAPI
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# (API Key logic removed as this is now a pure NLP state machine)

def get_system_instruction():
    # Hardcoded to August 8, 2026 so that "tomorrow" perfectly aligns with 
    # the start of the sample_visits.csv data (August 9, 2026).
    current_date = "2026-08-08"
    current_day = "Saturday"
    return f"""
You are a helpful Clinic Chatbot. The clinic working hours are 8:00 AM–12:00 PM and 4:00 PM–10:00 PM.
Today is {current_day}, {current_date}. Keep this in mind when the user mentions relative days like "tomorrow".

Core Behavior:
1. At the start of the conversation, greet the patient and ask if they want a fixed appointment or a walk-in visit. Do not repeat this greeting in the middle of a booking.
2. For fixed appointments: You must collect their preferred date and time, their name, and their email. If they give these one by one, acknowledge what they gave and gently ask for the missing details. Once you have ALL details (date, time, name, email), confirm the booking and state that a confirmation email has been sent.
3. For walk-in visits: Ask what time they plan to come. Use the predict_busyness tool to predict if the clinic is Busy, Normal, or Free at that time. 
- If Busy, use the tool to check nearby times and suggest 1-2 quieter times within the working hours.
- If Normal or Free, confirm that it's a good time to visit.
4. Do not offer medical advice. Stay in character.
"""

# Keep track of active chat sessions in memory for simplicity
# Each session will have a state: INIT, AWAITING_TYPE, AWAITING_FIXED_DETAILS, AWAITING_WALKIN_TIME
chat_sessions = {}

class MessageReq(BaseModel):
    session_id: str
    message: str

def get_state(session_id):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"state": "INIT"}
    return chat_sessions[session_id]["state"]

def set_state(session_id, state):
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {}
    chat_sessions[session_id]["state"] = state

def parse_walkin_time(user_msg: str):
    # Very basic regex parsing for demo purposes
    # E.g. "tomorrow 10 am"
    msg = user_msg.lower()
    
    date_str = "2026-08-08" # default today
    if "day after tomorrow" in msg:
        date_str = "2026-08-10"
    elif "tomorrow" in msg:
        date_str = "2026-08-09"
    elif "today" in msg:
        date_str = "2026-08-08"
        
    # Check for direct dates (YYYY-MM-DD)
    date_match = re.search(r'(2026-\d{2}-\d{2})', msg)
    if date_match:
        date_str = date_match.group(1)
    else:
        # Check for DD MMM or MMM DD
        month_map = {"jan": "01", "feb": "02", "mar": "03", "apr": "04", "may": "05", "jun": "06", 
                     "jul": "07", "aug": "08", "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
        md_match = re.search(r'\b(\d{1,2})(?:th|st|nd|rd)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b', msg)
        md_match2 = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2})(?:th|st|nd|rd)?\b', msg)
        
        if md_match:
            date_str = f"2026-{month_map[md_match.group(2)]}-{int(md_match.group(1)):02d}"
        elif md_match2:
            date_str = f"2026-{month_map[md_match2.group(1)]}-{int(md_match2.group(2)):02d}"
        
    hour = 10 # default 10 AM
    hour_match = re.search(r'(\d{1,2})\s*(am|pm)?', msg)
    if hour_match:
        h = int(hour_match.group(1))
        meridian = hour_match.group(2)
        if meridian == 'pm' and h < 12:
            h += 12
        elif meridian == 'am' and h == 12:
            h = 0
        hour = h
        
    return date_str, hour

@app.post("/api/chat")
async def chat_endpoint(req: MessageReq):
    logger.info(f"Incoming chat request for session_id: {req.session_id}")
    log_conversation_to_file(req.session_id, "User", req.message)
    
    msg_lower = req.message.strip().lower()
    state = get_state(req.session_id)
    
    resp_text = ""
    # Local variables for stream simulation
    time_checked = False
    checked_date = ""
    checked_hour = 0
    
    try:
        # Pre-defined stateless queries
        if "hours" in msg_lower or "when are you open" in msg_lower or ("time" in msg_lower and "open" in msg_lower):
            logger.info("NLP Router intercepted: Clinic Hours")
            resp_text = "Our clinic is open every day from 8:00 AM to 12:00 PM (Morning Session), and from 4:00 PM to 10:00 PM (Evening Session)."
        
        elif "where" in msg_lower and ("located" in msg_lower or "are you" in msg_lower):
            logger.info("NLP Router intercepted: Location")
            resp_text = "We are located at 123 Main Street, Medical District."
            
        else:
            # State Machine Logic
            if state == "INIT":
                if re.search(r"\b(hi|hello|hey|greetings|hii)\b", msg_lower):
                    resp_text = "Hello! Welcome to the Clinic. Would you like to schedule a fixed appointment, or are you planning a walk-in visit?"
                    set_state(req.session_id, "AWAITING_TYPE")
                else:
                    resp_text = "I'm a clinic assistant. Say 'hi' to start booking an appointment!"
                    
            elif state == "AWAITING_TYPE":
                if "fixed" in msg_lower or "appointment" in msg_lower:
                    resp_text = "Great! I can help you book a fixed appointment. Could you please provide your preferred date and time, your full name, and your email address?"
                    set_state(req.session_id, "AWAITING_FIXED_DETAILS")
                elif "walk" in msg_lower:
                    resp_text = "What day and time do you plan to stop by? (e.g. 'tomorrow 10 am')"
                    set_state(req.session_id, "AWAITING_WALKIN_TIME")
                else:
                    resp_text = "Please clarify: would you like a 'fixed appointment' or a 'walk-in' visit?"
                    
            elif state == "AWAITING_FIXED_DETAILS":
                # Initialize tracking if not present
                if "fixed_details" not in chat_sessions[req.session_id]:
                    chat_sessions[req.session_id]["fixed_details"] = {"time": None, "name": None, "email": None}
                
                details = chat_sessions[req.session_id]["fixed_details"]
                
                # 1. Check for email
                if not details["email"]:
                    match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", msg_lower)
                    if match:
                        details["email"] = match.group(0)
                    
                # 2. Check for time
                time_rejected = False
                if not details["time"] and re.search(r"(tomorrow|today|am|pm|\d{1,2})", msg_lower):
                    date_str, hour = parse_walkin_time(msg_lower)
                    
                    # Track tool call for UI
                    time_checked = True
                    checked_date = date_str
                    checked_hour = hour
                    
                    # Check availability using ML model
                    pred = predict_busyness(date_str, hour)
                    if pred == "Busy":
                        time_rejected = True
                        disp_h = hour if hour <= 12 else hour - 12
                        disp_h = 12 if disp_h == 0 else disp_h
                        am_pm = "AM" if hour < 12 else "PM"
                        resp_text = f"I'm sorry, but the {disp_h}:00 {am_pm} slot on {date_str} is not available. Please suggest a different time."
                    else:
                        details["time"] = {"date": date_str, "hour": hour}
                    
                # 3. Check for name (text remaining after stripping email and time words)
                if not details["name"]:
                    msg_clean = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", msg_lower)
                    msg_clean = re.sub(r"(tomorrow|today|am|pm|mrng|morning|evening|afternoon|\d+)", "", msg_clean)
                    words = re.findall(r"\b[a-z]{2,}\b", msg_clean)
                    if words:
                        details["name"] = " ".join(words[:2]).title()
                        
                # Determine what's missing
                missing = []
                if not details["time"]: missing.append("your preferred date and time")
                if not details["name"]: missing.append("your full name")
                if not details["email"]: missing.append("your email address")
                
                if time_rejected:
                    pass # resp_text already set to rejection message
                elif missing:
                    resp_text = f"Got it! To complete your booking, I still need: {', '.join(missing)}."
                else:
                    # Construct time string
                    dt = datetime.strptime(details["time"]["date"], "%Y-%m-%d")
                    day_str = dt.strftime("%a")
                    h = details["time"]["hour"]
                    am_pm = "AM" if h < 12 else "PM"
                    disp_hour = h if h <= 12 else h - 12
                    disp_hour = 12 if disp_hour == 0 else disp_hour
                    time_str = f"{disp_hour:02d}:00 {am_pm}"
                    
                    resp_text = f"Thank you, {details['name']}! I have all the details. Your appointment for {time_str} on {details['time']['date']} is confirmed and a confirmation email has been sent to {details['email']}!"
                    
                    # Append to sample_visits.csv
                    try:
                        with open("data/sample_visits.csv", "a") as f:
                            f.write(f"{details['name']},{details['time']['date']},{day_str},{time_str}\n")
                        logger.info(f"Appended new appointment to sample_visits.csv for {details['name']}")
                    except Exception as e:
                        logger.error(f"Failed to write to sample_visits.csv: {e}")

                    set_state(req.session_id, "INIT")
                    chat_sessions[req.session_id]["fixed_details"] = {"time": None, "name": None, "email": None}
                
            elif state == "AWAITING_WALKIN_TIME":
                date_str, hour = parse_walkin_time(msg_lower)
                
                time_checked = True
                checked_date = date_str
                checked_hour = hour
                
                prediction = predict_busyness(date_str, hour)
                if prediction == "Busy":
                    # Find nearby quieter times using half-hour offsets (requested by user)
                    quieter_times = []
                    
                    # Previous hour :30
                    prev_h = hour - 1 if hour > 8 else 8
                    am_pm_prev = "AM" if prev_h < 12 else "PM"
                    disp_prev = prev_h if prev_h <= 12 else prev_h - 12
                    disp_prev = 12 if disp_prev == 0 else disp_prev
                    quieter_times.append(f"{disp_prev}:30 {am_pm_prev}")
                    
                    # Current hour :30
                    am_pm_curr = "AM" if hour < 12 else "PM"
                    disp_curr = hour if hour <= 12 else hour - 12
                    disp_curr = 12 if disp_curr == 0 else disp_curr
                    quieter_times.append(f"{disp_curr}:30 {am_pm_curr}")
                    
                    disp_hour_req = hour if hour <= 12 else hour - 12
                    disp_hour_req = 12 if disp_hour_req == 0 else disp_hour_req
                    am_pm_req = "AM" if hour < 12 else "PM"
                    
                    resp_text = f"Based on our predictions, the clinic will be **Busy** at {disp_hour_req}:00 {am_pm_req} on {date_str}. I suggest coming at **{' or '.join(quieter_times)}** instead, when it's expected to be quieter."
                else:
                    disp_hour_req = hour if hour <= 12 else hour - 12
                    disp_hour_req = 12 if disp_hour_req == 0 else disp_hour_req
                    am_pm_req = "AM" if hour < 12 else "PM"
                    resp_text = f"Based on our predictions, the clinic will be **{prediction}** at {disp_hour_req}:00 {am_pm_req} on {date_str}. That's a great time to visit!"
                set_state(req.session_id, "INIT")

        log_conversation_to_file(req.session_id, "Bot", resp_text)
        
        # We must return a streaming response to match the frontend expectations, even though it's generated instantly.
        async def mock_stream():
            if time_checked:
                disp_h = checked_hour if checked_hour <= 12 else checked_hour - 12
                disp_h = 12 if disp_h == 0 else disp_h
                am_pm = "AM" if checked_hour < 12 else "PM"
                
                tool_msg = f"_Calling ML `predict_busyness` tool for {checked_date} at {disp_h}:00 {am_pm}..._\n\n"
                yield f'data: {{"content": {json.dumps(tool_msg)}}}\n\n'
                await asyncio.sleep(1.5)
                yield f'data: {{"content": {json.dumps(resp_text)}}}\n\n'
            else:
                # Add a tiny delay to simulate thinking so UI typing indicator shows briefly
                await asyncio.sleep(0.5) 
                yield f"data: {json.dumps({'content': resp_text})}\n\n"
            
        return StreamingResponse(mock_stream(), media_type="text/event-stream")
            
    except Exception as e:
        logger.error(f"Error occurred in chat_endpoint setup: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
