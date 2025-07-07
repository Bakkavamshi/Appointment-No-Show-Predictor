# 📅 Appointment No-Show Predictor

This project uses machine learning to predict whether a patient will show up for a medical appointment. It also recommends intervention strategies (like SMS reminders or phone calls) for high-risk cases — helping hospitals reduce wasted time and improve resource planning.

---

## 🚀 Project Objectives

- Predict patient no-shows using appointment history and demographics
- Analyze patterns using time, health, and location data
- Provide actionable interventions for high-risk cases
- Export final results with no-show probability and suggested strategy

---

## 🧠 Technologies Used

- Python
- Pandas
- Scikit-learn
- Random Forest Classifier
- Excel I/O using OpenPyXL

---

## 📊 Dataset Overview

- Source: Kaggle - Medical Appointment No Shows
- File: `appointments.xlsx`
- Key features:
  - Gender, Age, Scholarship, Neighborhood
  - Appointment and Scheduling Dates
  - Target: `No-show` (Yes or No)

---

## ⚙️ How to Run

### 🔧 Install Dependencies

```bash
pip install -r requirements.txt
