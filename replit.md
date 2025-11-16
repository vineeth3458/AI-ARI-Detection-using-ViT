# AI-Powered Acute Respiratory Infection (ARI) Detection System

## Project Overview

This is a comprehensive medical imaging web application designed for healthcare professionals to analyze chest X-rays and CT scans for Acute Respiratory Infections using advanced AI vision technology.

**Last Updated:** November 16, 2025

## System Architecture

### Core Components

1. **Medical AI Analyzer** (`medical_ai_analyzer.py`)
   - Integrates OpenAI GPT-5 Vision API for real medical image analysis
   - Provides detailed radiological findings and diagnostic predictions
   - Generates attention heatmaps for model interpretability
   - Supports fallback mode when API is not configured

2. **Database Layer** (`database.py`)
   - PostgreSQL database for persistent storage
   - SQLAlchemy ORM for data management
   - Stores analysis history, training runs, and model metrics
   - Supports cross-session data persistence

3. **PDF Report Generator** (`pdf_report_generator.py`)
   - Professional medical report generation using ReportLab
   - Individual and batch report capabilities
   - Includes patient metadata, findings, and recommendations

4. **Main Application** (`app.py`)
   - Streamlit-based web interface
   - Three main tabs: Image Analysis, Model Training, Performance Metrics
   - Real-time analysis and batch processing modes

## Key Features

### Image Analysis
- **Single Image Analysis**: Upload and analyze individual X-rays/CT scans
- **Batch Processing**: Analyze multiple images simultaneously
- **DICOM Support**: Full DICOM metadata extraction and display
- **Supported Formats**: JPEG, PNG, DICOM (.dcm)

### Diagnostic Capabilities
- Multi-class classification: Normal, Pneumonia, COVID-19, Other ARI
- Confidence scores and probability distributions
- Detailed radiological findings
- Key feature identification
- Clinical recommendations
- AI attention heatmaps for interpretability

### Report Generation
- Professional PDF reports for individual analyses
- Batch analysis reports with summary statistics
- Downloadable documentation for medical records

### Training Pipeline
- Configurable hyperparameters (epochs, batch size, learning rate)
- Advanced data augmentation options:
  - Random rotation with adjustable degrees
  - Horizontal and vertical flipping
  - Brightness and contrast adjustment
  - Random zoom
  - Gaussian noise injection
- Training history tracking
- Model comparison across different configurations

### Performance Metrics
- Real-time statistics based on actual analyses
- Diagnosis distribution visualizations
- Confidence score histograms
- Daily analysis volume trends
- Model comparison tools

## Database Schema

### Tables

1. **analyses**
   - Stores all image analysis results
   - Fields: filename, prediction, confidence, model, findings, key_features, recommendations, probabilities
   
2. **model_metrics**
   - Tracks model performance over time
   - Fields: model_name, accuracy, precision, recall, f1_score, confusion_matrix
   
3. **training_runs**
   - Records all training sessions
   - Fields: model_name, epochs, batch_size, learning_rate, final_train_accuracy, final_val_accuracy, training_metrics, augmentation_config

## API Integration

### OpenAI GPT-5 Vision
- **Purpose**: Real medical image analysis
- **Configuration**: Requires `OPENAI_API_KEY` environment variable
- **Model**: GPT-5 with vision capabilities (latest as of Aug 2025)
- **Fallback**: System operates in limited mode if API key not configured

## Environment Variables

Required:
- `OPENAI_API_KEY`: OpenAI API key for vision analysis
- `DATABASE_URL`: PostgreSQL connection string (auto-configured)

Auto-configured by Replit:
- `PGHOST`, `PGPORT`, `PGUSER`, `PGPASSWORD`, `PGDATABASE`

## Technology Stack

### Backend
- Python 3.11
- Streamlit 1.51.0+
- SQLAlchemy 2.0.44
- OpenAI 2.8.0
- PyDICOM 3.0.1

### Frontend
- Streamlit components
- Plotly for interactive visualizations
- Custom CSS for medical-grade UI

### Data Processing
- NumPy, Pandas
- Pillow (PIL) for image processing
- SciPy for scientific computing
- scikit-learn for metrics

### Reporting
- ReportLab 4.4.4 for PDF generation

## Usage Guide

### For Medical Professionals

1. **Single Image Analysis**
   - Select "Single Image Analysis" mode
   - Upload a chest X-ray or CT scan
   - Click "Analyze Image"
   - Review diagnostic results, findings, and recommendations
   - Download PDF report if needed

2. **Batch Processing**
   - Select "Batch Processing" mode
   - Upload multiple medical images
   - Click "Analyze All Images"
   - Review batch results and statistics
   - Download comprehensive batch report

3. **Training Custom Models**
   - Navigate to "Model Training" tab
   - Configure hyperparameters
   - Select data augmentation options
   - Upload custom dataset or use synthetic data
   - Monitor training progress
   - Compare model performance

4. **Performance Monitoring**
   - View real-time statistics
   - Analyze diagnosis distributions
   - Track confidence trends
   - Monitor daily analysis volume

## File Structure

```
.
├── app.py                      # Main Streamlit application
├── medical_ai_analyzer.py      # AI vision integration
├── database.py                 # Database models and operations
├── pdf_report_generator.py     # PDF report generation
├── pyproject.toml             # Python dependencies
├── .streamlit/
│   └── config.toml            # Streamlit configuration
└── replit.md                  # This documentation file
```

## System Status Indicators

The sidebar displays:
- **AI Model**: Active model name or "Fallback Mode"
- **API Status**: 🟢 Connected / 🔴 Not Configured
- **Database**: 🟢 Connected / 🔴 Not Connected
- **Total Analyses**: Count of all completed analyses

## Important Disclaimers

⚠️ **Medical Use Notice**: This tool is for research and educational purposes only. It should not be used as the sole basis for clinical decision-making. All diagnoses must be confirmed by qualified medical professionals including radiologists and physicians.

⚠️ **AI Limitations**: The AI system's predictions are probabilistic and may contain errors. Always consult with appropriate medical experts before making any treatment decisions.

## Development Notes

### Recent Changes (Nov 2025)
- Integrated OpenAI GPT-5 Vision for authentic medical image analysis
- Added PostgreSQL database for persistent storage
- Implemented batch processing with progress tracking
- Created professional PDF report generation
- Enhanced data augmentation options
- Added model comparison tools
- Updated performance metrics to derive from real data
- Improved UI with medical-grade color scheme

### Known Limitations
- DICOM batch processing not yet supported (only JPEG/PNG for batch mode)
- Training pipeline simulates metrics (real fine-tuning requires ML infrastructure)
- Performance metrics are demonstrations (require labeled validation data for true evaluation)

## Future Enhancements

Planned features:
- Integration with PACS systems
- Advanced Grad-CAM visualizations
- Multi-language support for reports
- Real-time collaboration features
- Enhanced DICOM support in batch mode
- Integration with electronic health record systems

## Support and Maintenance

For issues or questions:
1. Check system status indicators in sidebar
2. Verify OpenAI API key configuration
3. Ensure database connection is active
4. Review error messages in the interface

## License and Attribution

This application uses:
- OpenAI GPT-5 Vision API
- Replit PostgreSQL database
- Open-source Python libraries

Developed for medical professionals as a diagnostic assistance tool powered by advanced AI vision technology.
