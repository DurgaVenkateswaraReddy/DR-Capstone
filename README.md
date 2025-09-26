# DR Screening Web App (Flask + MongoDB)

Features
- Users: Patient and Doctor registration/login (hashed passwords).
- Patients: submit features, get prediction, view their reports, download one-page PDF.
- Doctors: view all patients’ reports, sort by probability (high→low or low→high), download PDFs.
- ML: Loads best model and threshold from artifacts/ and models/ created by train.py.

Prereqs
- Python 3.9+ (tested on Windows)
- MongoDB running locally or Atlas (provide MONGO_URI)
- Trained model artifacts created via train.py

Setup
1) Install deps:
   pip install -r requirements.txt

2) Train the models to generate artifacts and pickles:
   python train.py --data "C:\Users\durga\OneDrive\Desktop\DR full stack\DesiredDR.csv"

3) Set env vars (optional):
   set MONGO_URI=mongodb://localhost:27017/dr_app
   set SECRET_KEY=change-this

4) Run the app:
   python app.py
   Visit http://localhost:8000

Notes
- Ensure preprocessors.py exists so the custom transformer loads during unpickling.
- Reports collection schema:
  { patient_id, doctor_id, features, probability, prediction, threshold, model_name, created_at }
- This application is for research/educational use only.