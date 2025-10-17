# Absenteeism Risk Prediction Dashboard (Human-Centered AI)

This project provides a **responsible, fair and interpretable dashboard** to estimate the risk of employee absenteeism.  
Designed for **HR / team leaders** — not for data scientists — the interface explains predictions in plain language.

Developed as part of **CS698H — Human-Centered AI, IIT Kanpur**.

---

## 🚀 Deployment Links

| Component | Link |
|----------|------|
| **Frontend (Streamlit UI)** | [https://YOUR-HUGGINGFACE-SPACE](https://huggingface.co/spaces/chukey7277/absenteeism-fairness-ui) |
| **Backend (FastAPI API)** | [https://absenteeism-fairness-a3-1.onrender.com](https://absenteeism-fairness-a3-1.onrender.com) |
| **GitHub Repository** | [https://github.com/Chukey7277/absenteeism-fairness-a3](https://github.com/Chukey7277/absenteeism-fairness-a3) |

---

## 📌 How to Use (Without Local Setup)

You can directly use the UI hosted on Hugging Face:

1. **Ensure backend is live** — visit  
   https://absenteeism-fairness-a3-1.onrender.com  
   You should see something like:
   ```json
   {"detail":"Not Found"}

Open the frontend:
https://YOUR-HUGGINGFACE-SPACE

Enter employee features → Click Predict.

⚠️ If backend is OFF, HuggingFace UI will show a connection error.

💻 Running Locally (Alternative to Online Use)
1) Clone this repository
bash
Copy code
git clone https://github.com/Chukey7277/absenteeism-fairness-a3.git
cd absenteeism-fairness-a3
2) Start the Backend (FastAPI)
bash
Copy code
cd backend
pip install -r requirements.txt
uvicorn app:app --reload
Backend should be visible at: http://127.0.0.1:8000/docs

3) Start the Frontend (Streamlit)
Open a new terminal:

bash
Copy code
cd ../frontend
pip install -r requirements.txt
streamlit run ui_app.py
The UI will open at a local URL (usually http://localhost:8501)

## 🎯 What the Dashboard Provides
Human-friendly inputs (no raw codes)

Risk prediction as High / Low with probability

Threshold transparency (“How the model decides”)

Fairness details after mitigation

Performance summary in plain language

## ⚖️ Responsible Use Notice
This tool is designed for support — NOT for punishment.

Not for hiring or firing decisions

Not to discipline employees

Should always be combined with human judgment

## 📝 Project Credits
Member	Roll No
Tsewang Chukey	241110092
Lobsang Dhiki	251110045

Course: CS698H — Human-Centered AI (IIT Kanpur)

## 📜 License
Academic / educational use only. Not approved for commercial deployment.

