from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import os
import logging
import asyncio
import pickle
import pandas as pd
from datetime import datetime, timedelta
import json
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.tag import pos_tag
from nltk.stem import WordNetLemmatizer

# Download required NLTK data (runs once, silently)
for resource in ['punkt', 'punkt_tab', 'averaged_perceptron_tagger',
                  'averaged_perceptron_tagger_eng', 'wordnet']:
    nltk.download(resource, quiet=True)

logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("clinic_chatbot")

# ─────────────────────────────────────────────
# Load ML Model
# ─────────────────────────────────────────────
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

# ─────────────────────────────────────────────
# NLP setup
# ─────────────────────────────────────────────
lemmatizer = WordNetLemmatizer()

# Clinic valid hours (morning 8–11, evening 16–21)
MORNING_HOURS = list(range(8, 12))   # 8, 9, 10, 11
EVENING_HOURS = list(range(16, 22))  # 16, 17, 18, 19, 20, 21
VALID_HOURS = MORNING_HOURS + EVENING_HOURS

# ─────────────────────────────────────────────
# Helper utilities
# ─────────────────────────────────────────────
def log_conversation_to_file(session_id: str, role: str, content: str):
    try:
        os.makedirs("data", exist_ok=True)
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open("data/chat_logs.txt", "a", encoding="utf-8") as f:
            safe_content = content.replace('\n', ' ') if content else ''
            f.write(f"[{timestamp}] [Session: {session_id}] {role}: {safe_content}\n")
    except Exception as e:
        logger.error(f"Failed to write to chat log file: {e}")


def format_hour(hour: int) -> tuple:
    """Convert 24-hour int to (display_hour, 'AM'/'PM'). Fix #13: single helper."""
    am_pm = "AM" if hour < 12 else "PM"
    disp = hour if hour <= 12 else hour - 12
    disp = 12 if disp == 0 else disp
    return disp, am_pm


def is_valid_clinic_hour(hour: int) -> bool:
    """Fix #7: validate against real clinic schedule."""
    return hour in VALID_HOURS


# ─────────────────────────────────────────────
# ML Model wrapper
# ─────────────────────────────────────────────
def predict_busyness(date_str: str, hour: int) -> str:
    """
    Predicts Busy / Normal / Free for a given date and hour using the trained
    Random-Forest model. Clinic hours are 8–11 (Morning) and 16–21 (Evening).
    """
    if not clf:
        logger.warning("predict_busyness called but ML model is missing! Returning 'Normal'.")
        return "Normal"

    try:
        logger.info(f"🛠️  TOOL CALL: predict_busyness  date={date_str}  hour={hour}")
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
        logger.info(f"✅ TOOL RESULT: {date_str} {hour}:00 → {prediction}")
        return prediction
    except Exception as e:
        logger.error(f"predict_busyness error: {e}", exc_info=True)
        return "Unknown"


def find_quieter_slots(date_str: str, busy_hour: int) -> list:
    """
    Fix #5: Find ML-verified quieter alternatives around a busy hour.
    Searches ±1 then ±2 hours within valid clinic hours.
    Returns list of (hour, prediction) for non-Busy slots (max 2).
    """
    candidates = []
    for offset in [-1, 1, -2, 2]:
        candidate = busy_hour + offset
        if candidate in VALID_HOURS:
            pred = predict_busyness(date_str, candidate)
            if pred != "Busy":
                candidates.append((candidate, pred))
        if len(candidates) >= 2:
            break
    return candidates


# ─────────────────────────────────────────────
# NLP functions
# ─────────────────────────────────────────────
def parse_walkin_time(user_msg: str):
    """
    Parse date and hour from a user message.
    Fix #4: date portion is stripped from msg before the hour regex runs,
    so digits inside dates (e.g. '2026') are never mistaken for an hour.
    Returns (date_str, hour).
    """
    msg = user_msg.lower()
    today = datetime.now().date()

    # ── 1. Resolve relative day words ──
    date_str = today.strftime("%Y-%m-%d")
    if "day after tomorrow" in msg:
        date_str = (today + timedelta(days=2)).strftime("%Y-%m-%d")
    elif "tomorrow" in msg:
        date_str = (today + timedelta(days=1)).strftime("%Y-%m-%d")
    elif "today" in msg:
        date_str = today.strftime("%Y-%m-%d")

    msg_for_hour = msg  # will be narrowed after extracting date

    # ── 2. Look for explicit date patterns ──
    # YYYY-MM-DD
    yyyy_match = re.search(r'\d{4}-\d{2}-\d{2}', msg)
    if yyyy_match:
        date_str = yyyy_match.group(0)
        msg_for_hour = msg.replace(yyyy_match.group(0), " ")
    else:
        month_map = {
            "jan": "01", "feb": "02", "mar": "03", "apr": "04",
            "may": "05", "jun": "06", "jul": "07", "aug": "08",
            "sep": "09", "oct": "10", "nov": "11", "dec": "12"
        }
        # DD Mon / MonDD patterns
        md1 = re.search(
            r'\b(\d{1,2})(?:th|st|nd|rd)?\s*(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\b',
            msg
        )
        md2 = re.search(
            r'\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\s*(\d{1,2})(?:th|st|nd|rd)?\b',
            msg
        )
        if md1:
            date_str = f"{today.year}-{month_map[md1.group(2)]}-{int(md1.group(1)):02d}"
            msg_for_hour = msg.replace(md1.group(0), " ")
        elif md2:
            date_str = f"{today.year}-{month_map[md2.group(1)]}-{int(md2.group(2)):02d}"
            msg_for_hour = msg.replace(md2.group(0), " ")

    # ── 3. Parse hour from the date-stripped string ──
    hour = 10  # default 10 AM
    # Prefer explicit am/pm
    hm = re.search(r'\b(\d{1,2})\s*(am|pm)\b', msg_for_hour)
    if hm:
        h = int(hm.group(1))
        meridian = hm.group(2)
        if meridian == 'pm' and h < 12:
            h += 12
        elif meridian == 'am' and h == 12:
            h = 0
        hour = h
    else:
        # "at N" pattern without am/pm – infer session from value
        at_match = re.search(r'\bat\s+(\d{1,2})\b', msg_for_hour)
        if at_match:
            h = int(at_match.group(1))
            # Numbers 1-7 are ambiguous; assume PM for evening-range (4-10 PM)
            # Numbers 8-11 = morning; 1-7 assume PM (13-19)
            if 1 <= h <= 7:
                h += 12  # treat as PM
            hour = h

    return date_str, hour


def extract_name_nlp(text: str) -> str:
    """
    Extract person name using NLTK POS tagging.
    Fix #3: Looks for NNP/NNPS. If that fails (e.g. lowercase single words),
    uses a fallback for short conversational replies.
    """
    try:
        clean = re.sub(r"[\w\.-]+@[\w\.-]+\.\w+", "", text)
        
        # 1. Try strict NLTK proper noun extraction
        tokens = word_tokenize(clean)
        tagged = pos_tag(tokens)
        name_parts = [word.capitalize() for word, tag in tagged
                      if tag in ('NNP', 'NNPS') and len(word) > 1]
        if name_parts:
            return " ".join(name_parts[:3])  # max 3 words
            
        # 2. Fallback for short replies (e.g., "toffee", "that is the name toffee")
        clean_fallback = re.sub(
            r"\b(am|pm|today|tomorrow|day after tomorrow|morning|evening|afternoon|night|\d+|jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)\b",
            "", clean, flags=re.IGNORECASE)
        clean_fallback = re.sub(
            r"\b(my|name|is|that|this|the|it|yes|no|please|book|an|appointment|for|i|am|want|to|are|you|and|at|on|have|a)\b",
            "", clean_fallback, flags=re.IGNORECASE)
            
        words = [w.capitalize() for w in re.findall(r"\b[a-zA-Z]{2,}\b", clean_fallback)]
        
        # Only apply fallback if the user's message was relatively short
        if len(re.findall(r"\b[a-zA-Z]+\b", clean)) <= 7 and words:
            return " ".join(words[:3])
            
    except Exception as e:
        logger.warning(f"NLTK name extraction failed: {e}")
    return None


def detect_intent(text: str) -> str:
    """
    Fix #14: NLTK-lemmatized intent detection.
    Returns one of: 'greeting', 'hours', 'location', 'fixed', 'walkin', 'unknown'.
    """
    try:
        tokens = word_tokenize(text.lower())
        lemmas = {lemmatizer.lemmatize(t) for t in tokens}
        raw = text.lower()

        if lemmas & {"hi", "hello", "hey", "greet", "hii", "howdy"}:
            return "greeting"

        if (("hour" in lemmas or "time" in lemmas) and
                lemmas & {"open", "close", "work", "timing"}):
            return "hours"
        if "working hour" in raw or "clinic hour" in raw:
            return "hours"

        if "where" in raw and ("locat" in raw or "find" in raw or "address" in raw):
            return "location"

        fixed_words = {"fix", "fixed", "appointment", "book", "schedule", "reserve", "slot"}
        if lemmas & fixed_words:
            return "fixed"

        walkin_words = {"walk", "walkin", "drop", "come", "visit", "stop", "pass"}
        if lemmas & walkin_words:
            return "walkin"

    except Exception as e:
        logger.warning(f"detect_intent error: {e}")

    return "unknown"


def is_reset_request(text: str) -> bool:
    """Fix #15: detect cancel / restart requests."""
    try:
        tokens = word_tokenize(text.lower())
        lemmas = {lemmatizer.lemmatize(t) for t in tokens}
        reset_words = {"cancel", "restart", "reset", "stop", "quit", "exit", "start", "begin", "new"}
        return bool(lemmas & reset_words)
    except Exception:
        return False


# ─────────────────────────────────────────────
# FastAPI app
# ─────────────────────────────────────────────
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory session store
chat_sessions: dict = {}


class MessageReq(BaseModel):
    session_id: str
    message: str


def get_session(session_id: str) -> dict:
    if session_id not in chat_sessions:
        chat_sessions[session_id] = {"state": "INIT"}
    return chat_sessions[session_id]


def get_state(session_id: str) -> str:
    return get_session(session_id)["state"]


def set_state(session_id: str, state: str):
    get_session(session_id)["state"] = state


# ─────────────────────────────────────────────
# Chat endpoint
# ─────────────────────────────────────────────
@app.post("/api/chat")
async def chat_endpoint(req: MessageReq):
    logger.info(f"Incoming request  session={req.session_id}")
    log_conversation_to_file(req.session_id, "User", req.message)

    session = get_session(req.session_id)
    state = session["state"]
    msg_lower = req.message.strip().lower()

    resp_text = ""
    time_checked = False
    checked_date = ""
    checked_hour = 0
    extra_tool_calls: list = []   # list of (date, hour) for alternative slot checks

    try:
        # ── Global: Cancel / Restart (Fix #15) ──────────────────────────
        if is_reset_request(req.message) and state != "INIT":
            chat_sessions[req.session_id] = {"state": "INIT"}
            resp_text = ("No problem! Your request has been cancelled. "
                         "Say **'hi'** whenever you'd like to start again. 😊")

        # ── Stateless queries ────────────────────────────────────────────
        elif re.search(r'\bhours?\b', msg_lower) and re.search(r'\b(open|close|work|timing)\b', msg_lower):
            logger.info("NLP Router → Clinic Hours")
            resp_text = ("Our clinic is open every day:\n"
                         "🌅 **Morning:** 8:00 AM – 12:00 PM\n"
                         "🌆 **Evening:** 4:00 PM – 10:00 PM")

        elif "where" in msg_lower and re.search(r'\b(locat|find|address|are you)\b', msg_lower):
            logger.info("NLP Router → Location")
            resp_text = "We are located at **123 Main Street, Medical District**. 📍"

        # ── State Machine ────────────────────────────────────────────────
        else:
            if state == "INIT":
                # Fix #1: greeting handled fully by backend; frontend starts with empty chat
                intent = detect_intent(req.message)
                if intent == "greeting":
                    resp_text = ("Hello! 👋 Welcome to the Clinic Assistant.\n\n"
                                 "Would you like to schedule a **fixed appointment** "
                                 "(guaranteed slot) or are you planning a **walk-in visit**?")
                    set_state(req.session_id, "AWAITING_TYPE")
                else:
                    resp_text = ("Hi there! I'm the Clinic Assistant. "
                                 "Say **'hi'** to get started with booking! 🏥")

            elif state == "AWAITING_TYPE":
                intent = detect_intent(req.message)
                if intent == "fixed":
                    resp_text = ("Great! For a **fixed appointment**, please share your:\n"
                                 "- 📅 Preferred **date & time**\n"
                                 "- 👤 **Full name**\n"
                                 "- 📧 **Email address**\n\n"
                                 "Feel free to provide them all at once or one at a time!")
                    set_state(req.session_id, "AWAITING_FIXED_DETAILS")
                    session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}

                elif intent == "walkin":
                    resp_text = ("Sure! What **day and time** are you planning to stop by? "
                                 "(e.g. *'tomorrow 10 am'* or *'Monday 5 pm'*)")
                    set_state(req.session_id, "AWAITING_WALKIN_TIME")

                else:
                    resp_text = ("I didn't quite catch that. 😊 Would you like:\n"
                                 "- A **fixed appointment** (guaranteed slot), or\n"
                                 "- A **walk-in visit**?")

            elif state == "AWAITING_FIXED_DETAILS":
                if "fixed_details" not in session:
                    session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}
                details = session["fixed_details"]

                # 1. Extract email
                if not details["email"]:
                    email_match = re.search(r"[\w\.-]+@[\w\.-]+\.\w+", msg_lower)
                    if email_match:
                        details["email"] = email_match.group(0)

                # 2. Extract date & time (Fix #4: date stripped before hour regex)
                if not details["date"] and re.search(r"(tomorrow|today|am|pm|\d{1,2})", msg_lower):
                    date_str, hour = parse_walkin_time(msg_lower)
                    # Fix #7: validate clinic hours
                    if not is_valid_clinic_hour(hour):
                        disp_h, ap = format_hour(hour)
                        resp_text = (
                            f"I'm sorry, {disp_h}:00 {ap} is outside our working hours. "
                            "We're open **8:00 AM–12:00 PM** and **4:00 PM–10:00 PM**. "
                            "Please choose a time within those hours."
                        )
                    else:
                        # Fix #6: Fixed appointments are GUARANTEED — no ML busyness check
                        details["date"] = date_str
                        details["hour"] = hour

                # 3. Extract name using NLTK POS tagging (Fix #3)
                if not details["name"]:
                    extracted = extract_name_nlp(req.message)
                    if extracted:
                        details["name"] = extracted

                # Determine what's still missing
                missing = []
                if not details["date"]:
                    missing.append("your preferred **date & time**")
                if not details["name"]:
                    missing.append("your **full name**")
                if not details["email"]:
                    missing.append("your **email address**")

                if resp_text:
                    pass  # validation message already set above
                elif missing:
                    resp_text = f"Got it! I still need: {', '.join(missing)}."
                else:
                    # All details collected — confirm booking
                    disp_h, ap = format_hour(details["hour"])
                    time_str = f"{disp_h:02d}:00 {ap}"
                    dt = datetime.strptime(details["date"], "%Y-%m-%d")
                    day_full = dt.strftime("%A")
                    day_abbr = dt.strftime("%a")

                    resp_text = (
                        f"✅ **Appointment Confirmed!**\n\n"
                        f"👤 **Patient:** {details['name']}\n"
                        f"📅 **Date:** {day_full}, {details['date']}\n"
                        f"⏰ **Time:** {time_str}\n"
                        f"📧 **Confirmation sent to:** {details['email']}\n\n"
                        f"See you then! 😊"
                    )

                    # Append to CSV (Fix #11: header-aligned format)
                    try:
                        os.makedirs("data", exist_ok=True)
                        csv_path = "data/sample_visits.csv"
                        write_header = not os.path.exists(csv_path)
                        with open(csv_path, "a") as f:
                            if write_header:
                                f.write("Direct patients,Date,Day,Visit Time\n")
                            f.write(f"{details['name']},{details['date']},{day_abbr},{time_str}\n")
                        logger.info(f"Appointment saved for {details['name']}")
                    except Exception as e:
                        logger.error(f"Failed to write to sample_visits.csv: {e}")

                    set_state(req.session_id, "AWAITING_ANYTHING_ELSE")
                    session["fixed_details"] = None

            elif state == "AWAITING_WALKIN_TIME":
                date_str, hour = parse_walkin_time(msg_lower)

                # Fix #7: validate clinic hours
                if not is_valid_clinic_hour(hour):
                    disp_h, ap = format_hour(hour)
                    resp_text = (
                        f"I'm sorry, {disp_h}:00 {ap} is outside our working hours. "
                        "We're open **8:00 AM–12:00 PM** and **4:00 PM–10:00 PM**. "
                        "Please choose a time within those hours."
                    )
                else:
                    time_checked = True
                    checked_date = date_str
                    checked_hour = hour

                    prediction = predict_busyness(date_str, hour)
                    disp_h, ap = format_hour(hour)

                    if prediction == "Busy":
                        # Fix #5: suggest ML-verified quieter alternatives
                        alternatives = find_quieter_slots(date_str, hour)
                        extra_tool_calls = [(date_str, h) for h, _ in alternatives]

                        if alternatives:
                            alt_strs = []
                            for alt_h, alt_pred in alternatives:
                                alt_disp, alt_ap = format_hour(alt_h)
                                alt_strs.append(f"**{alt_disp}:00 {alt_ap}** ({alt_pred})")
                            resp_text = (
                                f"The clinic is expected to be **Busy** at {disp_h}:00 {ap} on {date_str}. "
                                f"I suggest coming at {' or '.join(alt_strs)} instead — much quieter! 🌿"
                            )
                        else:
                            resp_text = (
                                f"The clinic is expected to be **Busy** at {disp_h}:00 {ap} on {date_str}. "
                                "Nearby slots are also quite full. Consider an early morning visit "
                                "(**8:00 AM – 9:00 AM**) for the least wait time."
                            )
                    else:
                        resp_text = (
                            f"The clinic is expected to be **{prediction}** at {disp_h}:00 {ap} on {date_str}. "
                            "That's a great time to visit! 👍"
                        )

                    # Fix #2: Don't dump to INIT; ask if they need anything else
                    set_state(req.session_id, "AWAITING_ANYTHING_ELSE")

            elif state == "AWAITING_ANYTHING_ELSE":
                # Fix #2: graceful conversation end / loop
                yes_lemmas = {"yes", "yeah", "yep", "sure", "okay", "ok", "please", "more", "another"}
                try:
                    tokens = {lemmatizer.lemmatize(w.lower()) for w in word_tokenize(req.message)}
                except Exception:
                    tokens = set(msg_lower.split())

                intent = detect_intent(req.message)

                if tokens & yes_lemmas or intent in ("fixed", "walkin"):
                    if intent == "fixed":
                        resp_text = ("Of course! For a fixed appointment, I'll need your "
                                     "**date & time**, **full name**, and **email address**.")
                        set_state(req.session_id, "AWAITING_FIXED_DETAILS")
                        session["fixed_details"] = {"date": None, "hour": None, "name": None, "email": None}
                    elif intent == "walkin":
                        resp_text = "Sure! What **day and time** are you planning to stop by?"
                        set_state(req.session_id, "AWAITING_WALKIN_TIME")
                    else:
                        resp_text = ("How can I help? Would you like a **fixed appointment** "
                                     "or a **walk-in visit**?")
                        set_state(req.session_id, "AWAITING_TYPE")
                else:
                    resp_text = ("Thank you for choosing our clinic! Have a healthy day! 🌟\n"
                                 "Say **'hi'** anytime if you need help.")
                    set_state(req.session_id, "INIT")

        log_conversation_to_file(req.session_id, "Bot", resp_text)

        # ── Streaming response with ML tool-call annotations ─────────────
        async def mock_stream():
            if time_checked:
                disp_h, ap = format_hour(checked_hour)
                tool_msg = f"_🔍 Calling ML model: checking {checked_date} at {disp_h}:00 {ap}..._\n\n"
                yield f'data: {json.dumps({"content": tool_msg})}\n\n'
                await asyncio.sleep(1.2)

                for ec_date, ec_hour in extra_tool_calls:
                    ec_disp, ec_ap = format_hour(ec_hour)
                    ec_msg = f"_🔍 Checking alternative slot at {ec_disp}:00 {ec_ap}..._\n\n"
                    yield f'data: {json.dumps({"content": ec_msg})}\n\n'
                    await asyncio.sleep(0.6)

                yield f'data: {json.dumps({"content": resp_text})}\n\n'
            else:
                await asyncio.sleep(0.4)
                yield f'data: {json.dumps({"content": resp_text})}\n\n'

        return StreamingResponse(mock_stream(), media_type="text/event-stream")

    except Exception as e:
        logger.error(f"Error in chat_endpoint: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
