# Appointment No-Show Predictor (All-in-One Version)

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load Excel dataset
df = pd.read_excel('/content/appointments.xlsx')

# Convert date columns
df['ScheduledDay'] = pd.to_datetime(df['ScheduledDay'])
df['AppointmentDay'] = pd.to_datetime(df['AppointmentDay'])

# Feature engineering
df['DaysBetween'] = (df['AppointmentDay'] - df['ScheduledDay']).dt.days
df['Weekday'] = df['AppointmentDay'].dt.dayofweek
df['No-show'] = df['No-show'].map({'No': 0, 'Yes': 1})

le = LabelEncoder()
df['Gender'] = le.fit_transform(df['Gender'])
df['Neighbourhood'] = le.fit_transform(df['Neighbourhood'])

# Drop unnecessary columns
df.drop(['PatientId', 'AppointmentID', 'ScheduledDay', 'AppointmentDay'], axis=1, inplace=True)

# Train-test split
X = df.drop('No-show', axis=1)
y = df['No-show']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predict and evaluate
y_pred = model.predict(X_test)
print("Accuracy Score:", accuracy_score(y_test, y_pred))
print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Predict probabilities
y_prob = model.predict_proba(X_test)[:, 1]
risk_threshold = 0.7
high_risk_flags = (y_prob >= risk_threshold).astype(int)

# Create results DataFrame
risk_df = X_test.copy()
risk_df['Actual'] = y_test.values
risk_df['Predicted_Prob'] = y_prob
risk_df['High_Risk'] = high_risk_flags

# Define intervention logic
def recommend_intervention(prob):
    if prob >= 0.9:
        return "Phone call + Offer to reschedule"
    elif prob >= 0.7:
        return "Send SMS reminder"
    else:
        return "No special action"

risk_df['Intervention'] = risk_df['Predicted_Prob'].apply(recommend_intervention)

# Export to Excel
risk_df.to_excel('/content/high_risk_interventions.xlsx', index=False)
print("\n✅ All steps completed and results saved.")
