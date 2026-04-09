# 🏥 AI-Powered ARI Detection System

## 📌 Overview
The **AI-Powered Acute Respiratory Infection (ARI) Detection System** is a medical imaging web application that analyzes chest X-rays and CT scans using AI to detect respiratory diseases such as:

- Normal (Healthy)
- Pneumonia
- COVID-19
- Other ARI conditions

The system integrates **AI-based image analysis**, **database storage**, and **PDF report generation** into a single workflow for healthcare use.

---

## 🚀 Features

### 🔍 Image Analysis
- Upload medical images (JPEG, PNG, DICOM)
- AI-based diagnosis using vision models
- Multi-class classification (Normal, Pneumonia, COVID-19, ARI)
- Confidence scores and probability distribution
- Key findings and recommendations

### 📊 Dashboard & Visualization
- Interactive UI built with **Streamlit**
- Graphs using **Plotly**
- Performance metrics visualization
- Confusion matrix and classification reports

### 🗂️ Database Management
- Stores:
  - Analysis history
  - Predictions & probabilities
  - Model performance metrics
- Built with **SQLAlchemy ORM**
- Supports PostgreSQL (or fallback SQLite)

### 📄 PDF Report Generation
- Generates professional medical reports
- Includes:
  - Patient details
  - Diagnosis results
  - AI findings
- Built using **ReportLab**

### 🤖 AI Integration
- Uses **OpenAI GPT-5 Vision** for medical image analysis
- Fallback mode if API key is not available

---## 🏗️ Project Structure

-project/
│── app.py # Streamlit main app
│── main.py # Entry script
│── database.py # Database models & functions
│── medical_ai_analyzer.py # AI image analysis logic
│── pdf_report_generator.py # PDF report generation
│── airidetect.db # Local database (SQLite)
│── pyproject.toml # Dependencies
│── replit.md # Additional documentation


---



## 🏗️ Project Structure
