import os
import json
import logging
import asyncio
import pickle
import re
import pandas as pd
import requests as http_requests
from datetime import datetime, timedelta
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from dotenv import load_dotenv

load_dotenv()

# ─── LLM PROVIDER CONFIG ───
# Priority: OLLAMA (local, free, no rate limits) > GEMINI (cloud API)
# Set LLM_PROVIDER in .env to force one: "ollama" or "gemini"
# If not set, auto-detects: tries Ollama first, falls back to Gemini.

LLM_PROVIDER = os.getenv("LLM_PROVIDER", "auto").lower()
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3")
OLLAMA_URL = os.getenv("OLLAMA_URL", "http://localhost:11434")

# Check Ollama availability
ollama_available = False
try:
    r = http_requests.get(f"{OLLAMA_URL}/api/tags", timeout=2)
    if r.status_code == 200:
        ollama_available = True
except Exception:
    pass

# Setup Gemini if needed
gemini_available = False
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    import google.generativeai as genai
    genai.configure(api_key=api_key)
    gemini_available = True

# Resolve provider
if LLM_PROVIDER == "auto":
    if ollama_available:
        LLM_PROVIDER = "ollama"
    elif gemini_available:
        LLM_PROVIDER = "gemini"
    else:
        LLM_PROVIDER = "none"
elif LLM_PROVIDER == "gemini" and not gemini_available:
    print("WARNING: GEMINI_API_KEY not set! Falling back.")
    LLM_PROVIDER = "ollama" if ollama_available else "none"
elif LLM_PROVIDER == "ollama" and not ollama_available:
    print("WARNING: Ollama not running! Falling back.")
    LLM_PROVIDER = "gemini" if gemini_available else "none"

print(f"🤖 LLM Provider: {LLM_PROVIDER.upper()}" +
      (f" (model: {OLLAMA_MODEL})" if LLM_PROVIDER == "ollama" else ""))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
logger = logging.getLogger("clinic_chatbot")

# ─── ML MODEL SETUP ───
try:
    with open('model.pkl', 'rb') as f:
        data = pickle.load(f)
        clf = data['model']
        expected_cols = data['columns']
    logger.info("ML model loaded successfully.")
except Exception as e:
    logger.error(f"Error loading ML model: {e}")
    clf = None
    expected_cols = []

MORNING_HOURS = list(range(8, 12))
EVENING_HOURS = list(range(16, 22))
VALID_HOURS = MORNING_HOURS + EVENING_HOURS


def format_hour(hour: int) -> tuple:
    am_pm = "AM" if hour < 12 else "PM"
    disp = hour if hour <= 12 else hour - 12
    disp = 12 if disp == 0 else disp
    return disp, am_pm


def predict_busyness(date_str: str, hour: int) -> str:
    """Predicts Busy / Normal / Free for a given date and hour using the ML model."""
    hour = int(hour)
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
        return clf.predict(df)[0]
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        return "Unknown"


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


# ─── TIME PARSING ───
def parse_time(user_msg: str):
    """Parse date and hour from user message. Returns (date_str, hour)."""
    msg = user_msg.lower()
    today = datetime.now().date()

    date_str = today.strftime("%Y-%m-%d")
    if "day after tomorrow" in msg:
        date_str = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "tomorrow" in msg:
        date_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in msg:
        date_str = today.strftime("%Y-%m-%d")
    else:
        # Check for day-of-week names (e.g. "saturday", "monday")
        day_names = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                     "friday": 4, "saturday": 5, "sunday": 6,
                     "mon": 0, "tue": 1, "wed": 2, "thu": 3,
                     "fri": 4, "sat": 5, "sun": 6}
        for name, target_weekday in day_names.items():
            if name in msg:
                days_ahead = (target_weekday - today.weekday()) % 7
                if days_ahead == 0:
                    days_ahead = 7  # next week if same day
                date_str = (today + timedelta(days=days_ahead)).strftime("%Y-%m-%d")
                break

    msg_for_hour = msg
    yyyy_match = re.search(r'\d{4}-\d{2}-\d{2}', msg)
    if yyyy_match:
        date_str = yyyy_match.group(0)
        msg_for_hour = msg.replace(yyyy_match.group(0), " ")
    else:
        month_map = {"jan": "01", "feb": "02", "mar": "03", "apr": "04",
                     "may": "05", "jun": "06", "jul": "07", "aug": "08",
                     "sep": "09", "oct": "10", "nov": "11", "dec": "12"}
        md1 = re.search(r'\b(\d{1,2})(?:th|st|nd|rd)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)', msg)
        md2 = re.search(r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2})', msg)
        if md1:
            date_str = "{}-{}-{:02d}".format(today.year, month_map[md1.group(2)], int(md1.group(1)))
            msg_for_hour = msg.replace(md1.group(0), " ")
        elif md2:
            date_str = "{}-{}-{:02d}".format(today.year, month_map[md2.group(1)], int(md2.group(2)))
            msg_for_hour = msg.replace(md2.group(0), " ")

    hour = None
    hm = re.search(r'\b(\d{1,2})\s*(am|pm)\b', msg_for_hour)
    if hm:
        h = int(hm.group(1))
        if hm.group(2) == 'pm' and h < 12:
            h += 12
        elif hm.group(2) == 'am' and h == 12:
            h = 0
        hour = h
    else:
        at_match = re.search(r'\bat\s+(\d{1,2})\b', msg_for_hour)
        if at_match:
            h = int(at_match.group(1))
            if 1 <= h <= 7:
                h += 12
            hour = h
        else:
            bare = re.search(r'\b(\d{1,2})\b', msg_for_hour)
            if bare:
                h = int(bare.group(1))
                if 1 <= h <= 7:
                    h += 12
                hour = h

    return date_str, hour


# ─── LLM CALL (Ollama or Gemini) ───
def call_llm(prompt: str) -> str:
    """
    Routes the prompt to the configured LLM provider.
    Ollama: local, free, no rate limits.
    Gemini: cloud API, fast, but rate-limited on free tier.
    """
    if LLM_PROVIDER == "ollama":
        logger.info(f"🦙 Calling Ollama ({OLLAMA_MODEL})...")
        resp = http_requests.post(
            f"{OLLAMA_URL}/api/generate",
            json={"model": OLLAMA_MODEL, "prompt": prompt, "stream": False},
            timeout=60
        )
        resp.raise_for_status()
        return resp.json().get("response", "Sorry, I couldn't generate a response.")

    elif LLM_PROVIDER == "gemini":
        logger.info("✨ Calling Gemini API...")
        model = genai.GenerativeModel(model_name='gemini-2.5-flash')
        response = model.generate_content(prompt)
        return response.text or "Sorry, I couldn't generate a response."

    else:
        logger.warning("No LLM provider available. Returning fallback.")
        return "Sorry, no LLM is configured. Please set up Ollama or a Gemini API key."


# ─── FASTAPI APP ───
app = FastAPI()
app.add_middleware(
    CORSMiddleware, allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


class MessageReq(BaseModel):
    session_id: str
    message: str


# In-memory session store
chat_sessions = {}


def get_session(sid: str) -> dict:
    if sid not in chat_sessions:
        chat_sessions[sid] = {"state": "INIT"}
    return chat_sessions[sid]


@app.post("/api/chat")
async def chat_endpoint(req: MessageReq):
    session = get_session(req.session_id)
    state = session["state"]
    msg = req.message.strip()
    msg_lower = msg.lower()

    resp_text = ""
    use_llm = False
    llm_prompt = ""

    try:
        # ── CANCEL / RESTART ──
        cancel_words = {"cancel", "restart", "reset", "stop", "quit", "exit", "start over"}
        if any(w in msg_lower for w in cancel_words) and state != "INIT":
            chat_sessions[req.session_id] = {"state": "INIT"}
            resp_text = "No problem! Your request has been cancelled. Say **'hi'** whenever you'd like to start again. 😊"

        # ── INIT ──
        elif state == "INIT":
            greet_words = {"hi", "hello", "hey", "greetings", "hii", "howdy", "start"}
            # Allow skipping greeting if they mention their intent
            if any(w in msg_lower.split() for w in greet_words) or any(w in msg_lower for w in ["book", "appointment", "walk", "visit", "schedule", "fixed"]):
                # Fast track logic based on intent
                if "fixed" in msg_lower or "appointment" in msg_lower or "book" in msg_lower or "schedule" in msg_lower:
                    resp_text = ("Hello! For a **fixed appointment**, please share your:\n"
                                 "- 📅 Preferred **date & time**\n"
                                 "- 👤 **Full name**\n"
                                 "- 📧 **Email address**\n\n"
                                 "Feel free to provide them all at once or one at a time!")
                    session["state"] = "AWAITING_FIXED_DETAILS"
                    session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}
                elif "walk" in msg_lower or "visit" in msg_lower or "drop" in msg_lower or "come" in msg_lower:
                    resp_text = ("Hello! For a walk-in, what **day and time** are you planning to visit?\n"
                                 "(e.g. *'tomorrow 10 am'* or *'5 pm today'*)")
                    session["state"] = "AWAITING_WALKIN_TIME"
                else:
                    resp_text = ("Hello! 👋 Welcome to our Clinic.\n\n"
                                 "Would you like to schedule a **fixed appointment** (guaranteed slot) "
                                 "or are you planning a **walk-in visit**?")
                    session["state"] = "AWAITING_TYPE"
            else:
                resp_text = "Hi there! I'm the Clinic Assistant. How can I help you today? (You can say 'book an appointment' or 'walk-in')."

        # ── AWAITING_TYPE ──
        elif state == "AWAITING_TYPE":
            if "fixed" in msg_lower or "appointment" in msg_lower or "book" in msg_lower:
                resp_text = ("Great! For a **fixed appointment**, please share your:\n"
                             "- 📅 Preferred **date & time**\n"
                             "- 👤 **Full name**\n"
                             "- 📧 **Email address**\n\n"
                             "Feel free to provide them all at once or one at a time!")
                session["state"] = "AWAITING_FIXED_DETAILS"
                session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}
            elif "walk" in msg_lower or "visit" in msg_lower or "drop" in msg_lower or "come" in msg_lower:
                resp_text = ("Sure! What **day and time** are you planning to visit?\n"
                             "(e.g. *'tomorrow 10 am'* or *'5 pm today'*)")
                session["state"] = "AWAITING_WALKIN_TIME"
            else:
                resp_text = ("I didn't quite catch that. Would you like:\n"
                             "- A **fixed appointment** (guaranteed slot), or\n"
                             "- A **walk-in visit**?")

        # ── AWAITING_FIXED_DETAILS (all NLP, no LLM) ──
        elif state == "AWAITING_FIXED_DETAILS":
            if "fixed_details" not in session:
                session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}
            details = session["fixed_details"]

            # Email
            if not details["email"]:
                email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", msg)
                if email_match:
                    details["email"] = email_match.group(0)

            # Date & Time
            if not details["date"] and re.search(r"(tomorrow|today|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|am|pm|\d{1,2})", msg_lower):
                date_str, hour = parse_time(msg_lower)
                details["date"] = date_str
                if hour is not None:
                    if hour not in VALID_HOURS:
                        disp_h, ap = format_hour(hour)
                        resp_text = ("I'm sorry, **{}:00 {}** is outside our working hours. "
                                     "We're open **8 AM–12 PM** and **4 PM–10 PM**.").format(disp_h, ap)
                        details["date"] = None
                    else:
                        disp_h, ap = format_hour(hour)
                        time_str = "{:02d}:00 {}".format(disp_h, ap)
                        if get_booking_count(date_str, time_str) >= 2:
                            resp_text = ("That slot is fully booked. "
                                         "Please suggest a different time.")
                            details["date"] = None
                        else:
                            details["hour"] = hour

            # Name — fallback: strip emails, times, filler words; take remaining
            if not details["name"]:
                clean = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", msg)
                clean = re.sub(r"\b(am|pm|today|tomorrow|monday|tuesday|wednesday|thursday|friday|saturday|sunday|mon|tue|wed|thu|fri|sat|sun|\d+)\b", "", clean, flags=re.IGNORECASE)
                clean = re.sub(
                    r"\b(my|name|is|that|this|the|it|yes|no|please|book|an|appointment|for|"
                    r"i|want|to|are|you|and|at|on|have|a|hi|hello|fixed|email)\b",
                    "", clean, flags=re.IGNORECASE)
                words = [w.capitalize() for w in re.findall(r"\b[a-zA-Z]{2,}\b", clean)]
                if words and len(re.findall(r"\b[a-zA-Z]+\b", msg)) <= 7:
                    details["name"] = " ".join(words[:3])

            # Missing check
            missing = []
            if not details["date"] or details["hour"] is None:
                missing.append("your preferred **date & time (e.g. 10 AM or 4 PM)**")
            if not details["name"]:
                missing.append("your **full name**")
            if not details["email"]:
                missing.append("your **email address**")

            if resp_text:
                pass
            elif missing:
                resp_text = "Got it! I still need: {}.".format(", ".join(missing))
            else:
                disp_h, ap = format_hour(details["hour"])
                time_str = "{:02d}:00 {}".format(disp_h, ap)
                dt = datetime.strptime(details["date"], "%Y-%m-%d")
                day_full = dt.strftime("%A")
                day_abbr = dt.strftime("%a")

                resp_text = (
                    "✅ **Appointment Confirmed!**\n\n"
                    "👤 **Patient:** {}\n"
                    "📅 **Date:** {}, {}\n"
                    "⏰ **Time:** {}\n"
                    "📧 **Confirmation sent to:** {}\n\n"
                    "See you then! 😊"
                ).format(details["name"], day_full, details["date"], time_str, details["email"])

                # Save to CSV
                try:
                    os.makedirs("data", exist_ok=True)
                    csv_path = "data/sample_visits.csv"
                    write_header = not os.path.exists(csv_path)
                    with open(csv_path, "a") as f:
                        if write_header:
                            f.write("Direct patients,Date,Day,Visit Time\n")
                        f.write("{},{},{},{}\n".format(details["name"], details["date"], day_abbr, time_str))
                except Exception as e:
                    logger.error(f"CSV write error: {e}")

                session["state"] = "AWAITING_ANYTHING_ELSE"
                session["fixed_details"] = None

        # ── AWAITING_WALKIN_TIME (LLM USED HERE) ──
        elif state == "AWAITING_WALKIN_TIME":
            date_str, hour = parse_time(msg_lower)

            if hour is None:
                resp_text = ("Could you specify an exact hour? (e.g., '10 AM' or '4 PM'). "
                             "Our hours are **8 AM–12 PM** and **4 PM–10 PM**.")
            elif hour not in VALID_HOURS:
                disp_h, ap = format_hour(hour)
                resp_text = ("I'm sorry, **{}:00 {}** is outside our working hours. "
                             "We're open **8 AM–12 PM** and **4 PM–10 PM**. "
                             "Please choose a valid time.").format(disp_h, ap)
            else:
                # Run ML prediction
                prediction = predict_busyness(date_str, hour)
                disp_h, ap = format_hour(hour)

                # Also check nearby hours if busy
                nearby_info = ""
                if prediction == "Busy":
                    nearby = []
                    for offset in [-1, 1, -2, 2]:
                        c = hour + offset
                        if c in VALID_HOURS:
                            p = predict_busyness(date_str, c)
                            cd, ca = format_hour(c)
                            nearby.append("{}:00 {} → {}".format(cd, ca, p))
                    nearby_info = "Nearby slots: " + ", ".join(nearby)

                # 🔥 Use LLM to formulate a natural response based on ML data
                use_llm = True
                llm_prompt = (
                    "You are a professional clinic receptionist. Respond in a calm, helpful, clinical tone. "
                    "Do NOT use party emojis or overly enthusiastic language. This is a medical facility.\n\n"
                    "Patient wants to walk in at {}:00 {} on {} ({}).\n"
                    "ML Model Prediction for that slot: {}\n"
                    "{}\n\n"
                    "Rules:\n"
                    "- If Busy: politely inform them it will be crowded, and suggest 1-2 quieter nearby times from the data above.\n"
                    "- If Normal: let them know it should be a reasonable time with moderate wait.\n"
                    "- If Free: confirm it's a good, quiet time to visit.\n"
                    "- Keep the response to 2-3 sentences max. Be professional.\n"
                    "- CRITICAL: End with a statement. DO NOT ask any questions whatsoever. No question marks allowed."
                ).format(disp_h, ap, date_str, datetime.strptime(date_str, "%Y-%m-%d").strftime("%A"),
                         prediction, nearby_info)

                session["state"] = "AWAITING_ANYTHING_ELSE"

        # ── AWAITING_ANYTHING_ELSE ──
        elif state == "AWAITING_ANYTHING_ELSE":
            if re.search(r'\d|am|pm', msg_lower):
                # The user is confirming one of the suggested times!
                resp_text = ("Perfect, you are welcome to drop by then! Thank you for choosing our clinic. Have a healthy day! 🌟\n"
                             "Say **'hi'** anytime if you need help.")
                session["state"] = "INIT"
            else:
                end_words = {"no", "nothing", "thanks", "thank you", "nope", "bye", "goodbye", "ok", "okay", "fine", "yes", "sure", "yep"}
                if any(w in msg_lower.split() for w in end_words):
                    resp_text = ("Thank you for choosing our clinic! Have a healthy day! 🌟\n"
                                 "Say **'hi'** anytime if you need help.")
                    session["state"] = "INIT"
                else:
                    resp_text = ("How can I help? Would you like a **fixed appointment** "
                                 "or a **walk-in visit**?")
                    session["state"] = "AWAITING_TYPE"

        # ── RESPONSE ──
        async def generate():
            if use_llm:
                # Only LLM call in the entire app — for walk-in prediction interpretation
                try:
                    llm_text = await asyncio.to_thread(call_llm, llm_prompt)
                    # Append our state-machine question explicitly so the user isn't confused
                    llm_text += "\n\nIs there anything else I can help you with?"
                    yield f'data: {json.dumps({"content": llm_text})}\n\n'
                except Exception as e:
                    err = str(e)
                    if "429" in err:
                        yield f'data: {json.dumps({"content": "⏳ API rate limited. Please wait 30 seconds and try again."})}\n\n'
                    else:
                        yield f'data: {json.dumps({"content": "Error: " + err})}\n\n'
            else:
                await asyncio.sleep(0.3)
                yield f'data: {json.dumps({"content": resp_text})}\n\n'

        return StreamingResponse(generate(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Chat error: {e}", exc_info=True)
        async def generate_error():
            yield f'data: {json.dumps({"content": "Sorry, something went wrong. Please try again."})}\n\n'
        return StreamingResponse(generate_error(), media_type="text/event-stream")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
