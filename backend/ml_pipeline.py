import os
import pandas as pd
import numpy as np
import pickle
from datetime import datetime, timedelta
from sklearn.ensemble import RandomForestClassifier

def generate_synthetic_data(days=60):
    start_date = datetime.strptime("2026-08-01", "%Y-%m-%d")
    records = []
    
    np.random.seed(42) # For reproducibility
    
    for day_offset in range(days):
        current_date = start_date + timedelta(days=day_offset)
        day_of_week = current_date.strftime("%a")
        
        # Determine number of patients for the day
        # Weekends might be busier
        base_patients = 25 if day_of_week in ["Sat", "Sun"] else 15
        num_patients = np.random.poisson(base_patients)
        
        for _ in range(num_patients):
            # Pick session: morning or evening
            session = np.random.choice(["Morning", "Evening"], p=[0.4, 0.6])
            if session == "Morning":
                # 8:00 AM - 12:00 PM
                hour = np.random.randint(8, 12)
                minute = np.random.choice([0, 15, 30, 45])
            else:
                # 4:00 PM - 10:00 PM (16:00 - 22:00)
                hour = np.random.randint(16, 22)
                minute = np.random.choice([0, 15, 30, 45])
            
            time_str = f"{hour:02d}:{minute:02d}"
            
            records.append({
                "Date": current_date.strftime("%Y-%m-%d"),
                "Day": day_of_week,
                "Visit Time": time_str,
                "Hour": hour,
                "Session": session
            })
            
    df = pd.DataFrame(records)
    
    # Calculate busyness
    # Group by Date and Hour to find patient count per hour
    hourly_counts = df.groupby(['Date', 'Hour']).size().reset_index(name='PatientCount')
    
    # Assign labels based on patient count per hour
    def get_status(count):
        if count >= 4:
            return "Busy"
        elif count >= 2:
            return "Normal"
        else:
            return "Free"
            
    hourly_counts['Status'] = hourly_counts['PatientCount'].apply(get_status)
    
    # Merge back to assign status
    df = pd.merge(df, hourly_counts, on=['Date', 'Hour'])
    return df

def train_model():
    df = generate_synthetic_data(60)
    
    # Save the generated synthetic data to a CSV file
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/synthetic_visits.csv'
    df.to_csv(csv_path, index=False)
    print(f"Saved synthetic data to {csv_path}")
    
    # We want to predict status for any given hour, so we don't just train on individual visits,
    # but rather on the unique combinations of Date, Hour, Day, Session
    
    hourly_summary = df[['Date', 'Hour', 'Day', 'Session', 'Status']].drop_duplicates()
    
    # Features: Hour, Day (One-hot), Session (One-hot)
    X = hourly_summary[['Hour', 'Day', 'Session']]
    X = pd.get_dummies(X, columns=['Day', 'Session'])
    
    y = hourly_summary['Status']
    
    # We must ensure all days and sessions are present in dummies
    expected_cols = ['Hour', 'Day_Mon', 'Day_Tue', 'Day_Wed', 'Day_Thu', 'Day_Fri', 'Day_Sat', 'Day_Sun', 'Session_Morning', 'Session_Evening']
    for col in expected_cols:
        if col not in X.columns:
            X[col] = 0
            
    X = X[expected_cols]
    
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X, y)
    
    # Save model and expected columns
    with open('model.pkl', 'wb') as f:
        pickle.dump({'model': clf, 'columns': expected_cols}, f)
        
    print("Model trained and saved to model.pkl")
    print("Class distributions:")
    print(y.value_counts())
    
if __name__ == "__main__":
    train_model()
