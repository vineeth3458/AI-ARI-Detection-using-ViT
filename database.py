import os
from sqlalchemy import create_engine, Column, Integer, String, Float, DateTime, Text, JSON
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime
import streamlit as st

DATABASE_URL = os.environ.get("DATABASE_URL")

Base = declarative_base()

class Analysis(Base):
    __tablename__ = 'analyses'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    filename = Column(String(255), nullable=False)
    prediction = Column(String(50), nullable=False)
    confidence = Column(Float, nullable=False)
    model = Column(String(100), nullable=False)
    findings = Column(Text, nullable=True)
    key_features = Column(JSON, nullable=True)
    recommendations = Column(Text, nullable=True)
    probabilities = Column(JSON, nullable=False)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'filename': self.filename,
            'prediction': self.prediction,
            'confidence': self.confidence,
            'model': self.model,
            'findings': self.findings,
            'key_features': self.key_features,
            'recommendations': self.recommendations,
            'probabilities': self.probabilities
        }

class ModelMetrics(Base):
    __tablename__ = 'model_metrics'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100), nullable=False)
    accuracy = Column(Float, nullable=True)
    precision = Column(Float, nullable=True)
    recall = Column(Float, nullable=True)
    f1_score = Column(Float, nullable=True)
    confusion_matrix = Column(JSON, nullable=True)
    total_predictions = Column(Integer, default=0)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': self.model_name,
            'accuracy': self.accuracy,
            'precision': self.precision,
            'recall': self.recall,
            'f1_score': self.f1_score,
            'confusion_matrix': self.confusion_matrix,
            'total_predictions': self.total_predictions
        }

class TrainingRun(Base):
    __tablename__ = 'training_runs'
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    model_name = Column(String(100), nullable=False)
    epochs = Column(Integer, nullable=False)
    batch_size = Column(Integer, nullable=False)
    learning_rate = Column(Float, nullable=False)
    final_train_accuracy = Column(Float, nullable=True)
    final_val_accuracy = Column(Float, nullable=True)
    training_metrics = Column(JSON, nullable=True)
    augmentation_config = Column(JSON, nullable=True)
    
    def to_dict(self):
        return {
            'id': self.id,
            'timestamp': self.timestamp.strftime("%Y-%m-%d %H:%M:%S"),
            'model_name': self.model_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'learning_rate': self.learning_rate,
            'final_train_accuracy': self.final_train_accuracy,
            'final_val_accuracy': self.final_val_accuracy,
            'training_metrics': self.training_metrics,
            'augmentation_config': self.augmentation_config
        }

@st.cache_resource
def get_database_engine():
    if not DATABASE_URL:
        return None
    engine = create_engine(DATABASE_URL)
    Base.metadata.create_all(engine)
    return engine

@st.cache_resource
def get_session_maker():
    engine = get_database_engine()
    if not engine:
        return None
    return sessionmaker(bind=engine)

def get_db_session():
    SessionMaker = get_session_maker()
    if not SessionMaker:
        return None
    return SessionMaker()

def save_analysis(filename, prediction, confidence, model, findings=None, 
                 key_features=None, recommendations=None, probabilities=None):
    session = get_db_session()
    if not session:
        return None
    
    try:
        analysis = Analysis(
            filename=filename,
            prediction=prediction,
            confidence=confidence,
            model=model,
            findings=findings,
            key_features=key_features,
            recommendations=recommendations,
            probabilities=probabilities or {}
        )
        session.add(analysis)
        session.commit()
        result = analysis.to_dict()
        session.close()
        return result
    except Exception as e:
        session.rollback()
        session.close()
        return None

def get_all_analyses(limit=100):
    session = get_db_session()
    if not session:
        return []
    
    try:
        analyses = session.query(Analysis).order_by(
            Analysis.timestamp.desc()
        ).limit(limit).all()
        results = [a.to_dict() for a in analyses]
        session.close()
        return results
    except Exception as e:
        session.close()
        return []

def save_training_run(model_name, epochs, batch_size, learning_rate,
                     final_train_acc=None, final_val_acc=None,
                     training_metrics=None, augmentation_config=None):
    session = get_db_session()
    if not session:
        return None
    
    try:
        training_run = TrainingRun(
            model_name=model_name,
            epochs=epochs,
            batch_size=batch_size,
            learning_rate=learning_rate,
            final_train_accuracy=final_train_acc,
            final_val_accuracy=final_val_acc,
            training_metrics=training_metrics,
            augmentation_config=augmentation_config
        )
        session.add(training_run)
        session.commit()
        result = training_run.to_dict()
        session.close()
        return result
    except Exception as e:
        session.rollback()
        session.close()
        return None

def get_training_history(limit=20):
    session = get_db_session()
    if not session:
        return []
    
    try:
        runs = session.query(TrainingRun).order_by(
            TrainingRun.timestamp.desc()
        ).limit(limit).all()
        results = [r.to_dict() for r in runs]
        session.close()
        return results
    except Exception as e:
        session.close()
        return []

def update_model_metrics(model_name, accuracy=None, precision=None,
                        recall=None, f1_score=None, confusion_matrix=None,
                        total_predictions=0):
    session = get_db_session()
    if not session:
        return None
    
    try:
        metrics = ModelMetrics(
            model_name=model_name,
            accuracy=accuracy,
            precision=precision,
            recall=recall,
            f1_score=f1_score,
            confusion_matrix=confusion_matrix,
            total_predictions=total_predictions
        )
        session.add(metrics)
        session.commit()
        result = metrics.to_dict()
        session.close()
        return result
    except Exception as e:
        session.rollback()
        session.close()
        return None

def get_latest_metrics(model_name=None):
    session = get_db_session()
    if not session:
        return None
    
    try:
        query = session.query(ModelMetrics)
        if model_name:
            query = query.filter(ModelMetrics.model_name == model_name)
        
        metrics = query.order_by(ModelMetrics.timestamp.desc()).first()
        result = metrics.to_dict() if metrics else None
        session.close()
        return result
    except Exception as e:
        session.close()
        return None
