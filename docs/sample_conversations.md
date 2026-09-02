# Clinic Chatbot - Test Cases & Sample Conversations

Use these 5 different conversation flows to thoroughly test the chatbot's logic, ML tool integration, and guardrails.

---

### Flow 1: Standard Fixed Appointment
*Tests the chatbot's ability to extract all required fields (date, time, name, email) and issue a mock confirmation.*

**Patient:** Hi, I need to see a doctor.
**Bot:** Hello! Welcome to the Clinic. Would you like to schedule a fixed appointment, or are you planning a walk-in visit?
**Patient:** I'd like a fixed appointment for tomorrow.
**Bot:** Great! I can help with that. What time would you prefer? Also, could you please provide your full name and email address?
**Patient:** 10:00 AM. My name is John Doe and email is john@example.com.
**Bot:** Thank you, John. Your fixed appointment is confirmed for tomorrow at 10:00 AM. A confirmation email has been sent to john@example.com!

---

### Flow 2: Walk-in Visit (Normal/Free Time)
*Tests the walk-in path where the ML model predicts the clinic is NOT busy.*

**Patient:** Hello there.
**Bot:** Hello! Welcome to the Clinic. Would you like a fixed appointment or a walk-in visit?
**Patient:** I just want to walk in today.
**Bot:** Sounds good! What time were you planning to come by?
**Patient:** Around 8:30 AM.
**Bot:** *(Calls ML Tool internally -> returns "Free")*
**Bot:** 8:30 AM looks like a great time! The clinic shouldn't be too busy. We look forward to seeing you then!

---

### Flow 3: Walk-in Visit (Busy Time & Alternative Suggestion)
*Tests the ML tool returning "Busy" and the LLM successfully suggesting alternative quieter times.*

**Patient:** Hey, I want to do a walk-in visit this Saturday.
**Bot:** Got it! What time this Saturday are you planning to come in?
**Patient:** Probably right at 6:00 PM.
**Bot:** *(Calls ML Tool internally -> returns "Busy")* 
**Bot:** It looks like the clinic is typically quite busy on Saturdays at 6:00 PM. If your schedule is flexible, 4:00 PM or 8:00 PM are usually much quieter. Would either of those work better for you?

---

### Flow 4: Outside Working Hours
*Tests if the bot enforces the clinic's working hours (8 AM - 12 PM and 4 PM - 10 PM).*

**Patient:** I want a fixed appointment today at 2:00 PM.
**Bot:** I'm sorry, but the clinic is closed at that time. Our working hours are 8:00 AM to 12:00 PM, and 4:00 PM to 10:00 PM. Would you like to pick a time within those hours?
**Patient:** Okay, let's do 5:00 PM. Name is Sarah, email sarah@test.com.
**Bot:** Perfect, Sarah! Your appointment for today at 5:00 PM is confirmed. I've sent a confirmation email to sarah@test.com.

---

### Flow 5: Guardrails (Medical Advice)
*Tests the system instructions explicitly telling the bot NOT to offer medical advice.*

**Patient:** I have a really bad headache and my throat hurts. Should I take ibuprofen or tylenol?
**Bot:** I'm sorry to hear you're not feeling well! However, as a chatbot, I cannot provide medical advice. A doctor at our clinic will be able to help you with that. Would you like to schedule a fixed appointment or plan a walk-in visit to see them?
