🏥 AI-Powered ARI Detection System
📌 Overview

The AI-Powered Acute Respiratory Infection (ARI) Detection System is a medical imaging web application that analyzes chest X-rays and CT scans using AI to detect respiratory diseases such as:

Normal (Healthy)
Pneumonia
COVID-19
Other ARI conditions

The system integrates AI-based image analysis, database storage, and PDF report generation into a single workflow for healthcare use.

🚀 Features
🔍 Image Analysis
Upload medical images (JPEG, PNG, DICOM)
AI-based diagnosis using vision models
Multi-class classification (Normal, Pneumonia, COVID-19, ARI)
Confidence scores and probability distribution
Key findings and recommendations
📊 Dashboard & Visualization
Interactive UI built with Streamlit
Graphs using Plotly
Performance metrics visualization
Confusion matrix and classification reports
🗂️ Database Management
Stores:
Analysis history
Predictions & probabilities
Model performance metrics
Built with SQLAlchemy ORM
Supports PostgreSQL (or fallback)
📄 PDF Report Generation
Generates professional medical reports
Includes:
Patient details
Diagnosis results
AI findings
Built using ReportLab
🤖 AI Integration
Uses OpenAI GPT-5 Vision for medical image analysis
Fallback mode if API key is not available
🏗️ Project Structure
project/
│── app.py                     # Streamlit main app
│── main.py                    # Entry script (basic)
│── database.py                # Database models & functions
│── medical_ai_analyzer.py     # AI image analysis logic
│── pdf_report_generator.py    # PDF report generation
│── airidetect.db              # Local database (SQLite)
│── pyproject.toml             # Dependencies
│── replit.md                  # Project documentation
⚙️ Installation
1️⃣ Clone the repository
git clone <your-repo-url>
cd project
2️⃣ Create virtual environment
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
3️⃣ Install dependencies
pip install -r requirements.txt

(or use pyproject.toml if using modern tools like Poetry/UV)

🔑 Environment Variables

Create a .env file:

OPENAI_API_KEY=your_api_key_here
DATABASE_URL=your_database_url   # optional
▶️ Running the Application
streamlit run app.py

Then open:

http://localhost:8501
🧠 How It Works
User uploads an X-ray/CT scan
Image is processed and encoded
Sent to AI model for analysis
Model returns:
Diagnosis
Confidence
Findings
Results are:
Displayed on UI
Stored in database
Exported as PDF (optional)
🗄️ Database Schema
analyses
id
timestamp
filename
prediction
confidence
findings
recommendations
probabilities
model_metrics
accuracy
precision
recall
f1_score
confusion_matrix
📦 Tech Stack
Frontend: Streamlit
Backend: Python
AI Model: OpenAI GPT-5 Vision
Database: PostgreSQL / SQLite
ORM: SQLAlchemy
Visualization: Plotly
PDF Generation: ReportLab
Image Processing: PIL, NumPy
⚠️ Disclaimer

This system is intended for:

Educational purposes
Research assistance

❗ Not a replacement for professional medical diagnosis

🔮 Future Improvements
Custom trained deep learning model (CNN)
Real-time hospital integration
User authentication system
Cloud deployment (AWS/GCP)
Model explainability (Grad-CAM)
