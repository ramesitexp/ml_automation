"""
EURON ML Automation - FastAPI Backend
A comprehensive automated machine learning system
"""

import os
import json
import pickle
import uuid
import warnings
import math
from datetime import datetime
from typing import List, Dict, Any, Optional
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, Response
from pydantic import BaseModel
import uvicorn

# Suppress warnings
warnings.filterwarnings('ignore')


class NaNSafeJSONEncoder(json.JSONEncoder):
    """Custom JSON encoder that handles NaN, Infinity values"""
    def default(self, obj):
        if isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
        if isinstance(obj, np.floating):
            if np.isnan(obj) or np.isinf(obj):
                return 0.0
            return float(obj)
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)
    
    def encode(self, obj):
        return super().encode(self._sanitize(obj))
    
    def _sanitize(self, obj):
        if isinstance(obj, dict):
            return {k: self._sanitize(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [self._sanitize(item) for item in obj]
        elif isinstance(obj, float):
            if math.isnan(obj) or math.isinf(obj):
                return 0.0
            return obj
        elif isinstance(obj, (np.floating, np.float64, np.float32)):
            val = float(obj)
            if math.isnan(val) or math.isinf(val):
                return 0.0
            return val
        elif isinstance(obj, (np.integer, np.int64, np.int32)):
            return int(obj)
        elif isinstance(obj, np.ndarray):
            return self._sanitize(obj.tolist())
        return obj


def sanitize_for_json(obj):
    """Recursively sanitize an object to be JSON-safe"""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(item) for item in obj]
    elif isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return 0.0
        return obj
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        val = float(obj)
        if math.isnan(val) or math.isinf(val):
            return 0.0
        return val
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return sanitize_for_json(obj.tolist())
    return obj

# ML Libraries
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    mean_squared_error, mean_absolute_error, r2_score,
    confusion_matrix, classification_report, silhouette_score,
    davies_bouldin_score, calinski_harabasz_score
)

# Supervised Learning Models
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.svm import SVC, SVR
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# Unsupervised Learning Models
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering

# XGBoost
try:
    from xgboost import XGBClassifier, XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

# YData Profiling
from ydata_profiling import ProfileReport

app = FastAPI(
    title="ML Automation",
    description="Automated Machine Learning System by me12",
    version="1.0.0"
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Directory Setup
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
REPORTS_DIR = BASE_DIR / "reports"

for dir_path in [DATA_DIR, MODELS_DIR, REPORTS_DIR]:
    dir_path.mkdir(exist_ok=True)

# Global storage for session data
session_data = {}


# Pydantic Models
class DataInfo(BaseModel):
    session_id: str
    filename: str
    rows: int
    columns: int
    column_names: List[str]
    dtypes: Dict[str, str]
    missing_values: Dict[str, int]
    numeric_columns: List[str]
    categorical_columns: List[str]


class FeatureSelection(BaseModel):
    session_id: str
    features: List[str]
    target: Optional[str] = None
    problem_type: str  # 'classification', 'regression', 'clustering'


class TrainingConfig(BaseModel):
    session_id: str
    test_size: float = 0.2
    random_state: int = 42
    scale_features: bool = True
    handle_missing: str = "mean"  # 'mean', 'median', 'mode', 'drop'
    selected_models: Optional[List[str]] = None




class ModelResult(BaseModel):
    model_name: str
    metrics: Dict[str, float]
    training_time: float
    parameters: Dict[str, Any]


# Helper Functions
def detect_problem_type(y: pd.Series) -> str:
    """Detect if the problem is classification or regression"""
    unique_ratio = len(y.unique()) / len(y)
    if y.dtype == 'object' or unique_ratio < 0.05 or len(y.unique()) <= 20:
        return 'classification'
    return 'regression'


def get_column_info(df: pd.DataFrame) -> Dict:
    """Get detailed column information"""
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
    
    return {
        'numeric_columns': numeric_cols,
        'categorical_columns': categorical_cols,
        'dtypes': df.dtypes.astype(str).to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'unique_counts': {col: df[col].nunique() for col in df.columns}
    }


def preprocess_data(df: pd.DataFrame, features: List[str], target: Optional[str], 
                    handle_missing: str, scale_features: bool) -> tuple:
    """Preprocess the data for training"""
    X = df[features].copy()
    y = df[target].copy() if target else None
    
    # Separate numeric and categorical columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
    
    # Handle missing values for numeric columns
    if handle_missing == 'drop':
        X = X.dropna()
        if y is not None:
            y = y.loc[X.index]
    else:
        if numeric_cols:
            strategy = handle_missing if handle_missing in ['mean', 'median'] else 'mean'
            imputer = SimpleImputer(strategy=strategy)
            X[numeric_cols] = imputer.fit_transform(X[numeric_cols])
        
        if categorical_cols:
            imputer = SimpleImputer(strategy='most_frequent')
            X[categorical_cols] = imputer.fit_transform(X[categorical_cols])
    
    # Encode categorical columns
    label_encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col].astype(str))
        label_encoders[col] = le
    
    # Encode target if categorical
    target_encoder = None
    if y is not None and y.dtype == 'object':
        target_encoder = LabelEncoder()
        y = pd.Series(target_encoder.fit_transform(y.astype(str)), index=y.index)
    
    # Scale features
    scaler = None
    if scale_features and numeric_cols:
        scaler = StandardScaler()
        X[numeric_cols] = scaler.fit_transform(X[numeric_cols])
    
    return X, y, scaler, label_encoders, target_encoder


def get_classification_models() -> Dict:
    """Get all classification models"""
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'SVM Classifier': SVC(probability=True, random_state=42),
        'Decision Tree Classifier': DecisionTreeClassifier(random_state=42),
        'Random Forest Classifier': RandomForestClassifier(n_estimators=100, random_state=42),
        'KNN Classifier': KNeighborsClassifier(),
        'Gradient Boosting Classifier': GradientBoostingClassifier(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost Classifier'] = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
    return models


def get_regression_models() -> Dict:
    """Get all regression models"""
    models = {
        'Linear Regression': LinearRegression(),
        'Ridge Regression': Ridge(random_state=42),
        'Lasso Regression': Lasso(random_state=42),
        'SVR': SVR(),
        'Decision Tree Regressor': DecisionTreeRegressor(random_state=42),
        'Random Forest Regressor': RandomForestRegressor(n_estimators=100, random_state=42),
        'KNN Regressor': KNeighborsRegressor(),
        'Gradient Boosting Regressor': GradientBoostingRegressor(random_state=42),
    }
    if XGBOOST_AVAILABLE:
        models['XGBoost Regressor'] = XGBRegressor(random_state=42)
    return models


def get_clustering_models(n_clusters: int = 3) -> Dict:
    """Get all clustering models"""
    return {
        'KMeans': KMeans(n_clusters=n_clusters, random_state=42),
        'DBSCAN': DBSCAN(eps=0.5, min_samples=5),
        'Agglomerative Clustering': AgglomerativeClustering(n_clusters=n_clusters),
    }


def safe_float(value) -> float:
    """Convert value to float, replacing NaN/Inf with 0"""
    if value is None:
        return 0.0
    val = float(value)
    if np.isnan(val) or np.isinf(val):
        return 0.0
    return val


def evaluate_classification(y_true, y_pred, y_prob=None) -> Dict:
    """Evaluate classification model"""
    metrics = {
        'accuracy': safe_float(accuracy_score(y_true, y_pred)),
        'precision': safe_float(precision_score(y_true, y_pred, average='weighted', zero_division=0)),
        'recall': safe_float(recall_score(y_true, y_pred, average='weighted', zero_division=0)),
        'f1_score': safe_float(f1_score(y_true, y_pred, average='weighted', zero_division=0)),
    }
    return metrics


def evaluate_regression(y_true, y_pred) -> Dict:
    """Evaluate regression model"""
    return {
        'mse': safe_float(mean_squared_error(y_true, y_pred)),
        'rmse': safe_float(np.sqrt(mean_squared_error(y_true, y_pred))),
        'mae': safe_float(mean_absolute_error(y_true, y_pred)),
        'r2_score': safe_float(r2_score(y_true, y_pred)),
    }


def evaluate_clustering(X, labels) -> Dict:
    """Evaluate clustering model"""
    # Filter out noise points for DBSCAN
    mask = labels != -1
    if mask.sum() < 2 or len(np.unique(labels[mask])) < 2:
        return {
            'silhouette_score': -1.0,
            'davies_bouldin_score': -1.0,
            'calinski_harabasz_score': -1.0,
            'n_clusters': int(len(np.unique(labels[mask])) if mask.sum() > 0 else 0),
            'noise_points': int((~mask).sum())
        }
    
    return {
        'silhouette_score': safe_float(silhouette_score(X[mask], labels[mask])),
        'davies_bouldin_score': safe_float(davies_bouldin_score(X[mask], labels[mask])),
        'calinski_harabasz_score': safe_float(calinski_harabasz_score(X[mask], labels[mask])),
        'n_clusters': int(len(np.unique(labels[mask]))),
        'noise_points': int((~mask).sum())
    }


# API Endpoints
@app.get("/")
async def root():
    return {
        "message": "Welcome to EURON ML Automation API",
        "version": "1.0.0",
        "features": ["Automated ML Training", "EDA Reports", "Model Evaluation"]
    }


@app.get("/status")
async def get_status():
    """Get system status"""
    return {
        "backend": "running",
        "version": "1.0.0",
        "status": "active"
    }


@app.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    """Upload a dataset file"""
    session_id = str(uuid.uuid4())
    
    # Save file
    file_path = DATA_DIR / f"{session_id}_{file.filename}"
    
    try:
        contents = await file.read()
        with open(file_path, "wb") as f:
            f.write(contents)
        
        # Read the data
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file.filename.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        elif file.filename.endswith('.json'):
            df = pd.read_json(file_path)
        else:
            raise HTTPException(status_code=400, detail="Unsupported file format")
        
        # Store in session
        session_data[session_id] = {
            'df': df,
            'filename': file.filename,
            'file_path': str(file_path)
        }
        
        # Get column info
        col_info = get_column_info(df)
        
        return DataInfo(
            session_id=session_id,
            filename=file.filename,
            rows=len(df),
            columns=len(df.columns),
            column_names=df.columns.tolist(),
            dtypes=col_info['dtypes'],
            missing_values=col_info['missing_values'],
            numeric_columns=col_info['numeric_columns'],
            categorical_columns=col_info['categorical_columns']
        )
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{session_id}/preview")
async def preview_data(session_id: str, rows: int = 10):
    """Preview the uploaded data"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session_data[session_id]['df']
    return {
        'head': df.head(rows).to_dict(orient='records'),
        'tail': df.tail(rows).to_dict(orient='records'),
        'shape': df.shape,
        'columns': df.columns.tolist()
    }


@app.get("/data/{session_id}/statistics")
async def get_statistics(session_id: str):
    """Get statistical summary of the data"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session_data[session_id]['df']
    
    # Numeric statistics
    numeric_stats = df.describe().to_dict()
    
    # Categorical statistics
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    categorical_stats = {}
    for col in categorical_cols:
        categorical_stats[col] = {
            'unique': int(df[col].nunique()),
            'top': str(df[col].mode().iloc[0]) if not df[col].mode().empty else None,
            'freq': int(df[col].value_counts().iloc[0]) if not df[col].value_counts().empty else 0,
            'value_counts': df[col].value_counts().head(10).to_dict()
        }
    
    return {
        'numeric_statistics': numeric_stats,
        'categorical_statistics': categorical_stats,
        'correlation_matrix': df.select_dtypes(include=[np.number]).corr().to_dict() if len(df.select_dtypes(include=[np.number]).columns) > 0 else {}
    }


@app.post("/data/{session_id}/eda-report")
async def generate_eda_report(session_id: str, background_tasks: BackgroundTasks):
    """Generate comprehensive EDA report using YData Profiling"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session_data[session_id]['df']
    report_path = REPORTS_DIR / f"{session_id}_eda_report.html"
    
    try:
        # Generate report with minimal settings for speed
        profile = ProfileReport(
            df, 
            title="EURON ML Automation - EDA Report",
            explorative=True,
            minimal=False,
            progress_bar=False
        )
        profile.to_file(report_path)
        
        session_data[session_id]['eda_report'] = str(report_path)
        
        return {"message": "EDA report generated", "report_path": str(report_path)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/data/{session_id}/eda-report")
async def get_eda_report(session_id: str):
    """Get the generated EDA report"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if 'eda_report' not in session_data[session_id]:
        raise HTTPException(status_code=404, detail="EDA report not generated yet")
    
    report_path = session_data[session_id]['eda_report']
    return FileResponse(report_path, media_type='text/html')


@app.post("/train")
async def train_models(config: TrainingConfig):
    """Train multiple models and compare them"""
    if config.session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_data[config.session_id]
    df = session['df']
    
    if 'features' not in session or 'problem_type' not in session:
        raise HTTPException(status_code=400, detail="Please select features first")
    
    features = session['features']
    target = session.get('target')
    problem_type = session['problem_type']
    
    # Preprocess data
    X, y, scaler, label_encoders, target_encoder = preprocess_data(
        df, features, target, config.handle_missing, config.scale_features
    )
    
    results = []
    best_model = None
    best_score = -float('inf')
    
    if problem_type == 'clustering':
        # Clustering
        n_clusters = session.get('n_clusters', 3)
        models = get_clustering_models(n_clusters)
        
        if config.selected_models:
            models = {k: v for k, v in models.items() if k in config.selected_models}
        
        for name, model in models.items():
            try:
                start_time = datetime.now()
                labels = model.fit_predict(X)
                training_time = (datetime.now() - start_time).total_seconds()
                
                metrics = evaluate_clustering(X.values, labels)
                
                # Sanitize all values for JSON
                result = {
                    'model_name': name,
                    'metrics': sanitize_for_json(metrics),
                    'training_time': safe_float(training_time),
                    'parameters': sanitize_for_json(model.get_params()),
                    'labels': sanitize_for_json(labels.tolist())
                }
                results.append(result)
                
                # Best model based on silhouette score
                if metrics['silhouette_score'] > best_score:
                    best_score = metrics['silhouette_score']
                    best_model = {
                        'name': name,
                        'model': model,
                        'metrics': sanitize_for_json(metrics)
                    }
            except Exception as e:
                results.append({
                    'model_name': name,
                    'error': str(e),
                    'metrics': {},
                    'training_time': 0,
                    'parameters': {}
                })
    
    else:
        # Supervised Learning
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=config.test_size, random_state=config.random_state
        )
        
        if problem_type == 'classification':
            models = get_classification_models()
            if config.selected_models:
                models = {k: v for k, v in models.items() if k in config.selected_models}
            
            for name, model in models.items():
                try:
                    start_time = datetime.now()
                    model.fit(X_train, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    y_pred = model.predict(X_test)
                    y_prob = model.predict_proba(X_test) if hasattr(model, 'predict_proba') else None
                    
                    metrics = evaluate_classification(y_test, y_pred, y_prob)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='accuracy')
                    metrics['cv_mean'] = safe_float(cv_scores.mean())
                    metrics['cv_std'] = safe_float(cv_scores.std())
                    
                    # Confusion matrix
                    cm = confusion_matrix(y_test, y_pred)
                    
                    # Save confusion matrix
                    import matplotlib.pyplot as plt
                    import seaborn as sns
                    plt.figure(figsize=(8, 6))
                    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
                    plt.title(f'Confusion Matrix - {name}')
                    plt.ylabel('True Label')
                    plt.xlabel('Predicted Label')
                    cm_path = REPORTS_DIR / f"{config.session_id}_{name}_cm.png"
                    plt.savefig(cm_path)
                    plt.close()
                    
                    # Sanitize classification report for JSON
                    class_report = classification_report(y_test, y_pred, output_dict=True)
                    for key, value in class_report.items():
                        if isinstance(value, dict):
                            for k, v in value.items():
                                class_report[key][k] = safe_float(v)
                        elif isinstance(value, (int, float)):
                            class_report[key] = safe_float(value)
                    
                    result = {
                        'model_name': name,
                        'metrics': metrics,
                        'training_time': safe_float(training_time),
                        'parameters': model.get_params(),
                        'confusion_matrix': cm.tolist(),
                        'classification_report': class_report
                    }
                    results.append(result)
                    
                    # Best model based on accuracy
                    if metrics['accuracy'] > best_score:
                        best_score = metrics['accuracy']
                        best_model = {
                            'name': name,
                            'model': model,
                            'metrics': sanitize_for_json(metrics)
                        }
                
                except Exception as e:
                    results.append({
                        'model_name': name,
                        'error': str(e),
                        'metrics': {},
                        'training_time': 0,
                        'parameters': {}
                    })
        
        else:  # Regression
            models = get_regression_models()
            if config.selected_models:
                models = {k: v for k, v in models.items() if k in config.selected_models}
            
            for name, model in models.items():
                try:
                    start_time = datetime.now()
                    model.fit(X_train, y_train)
                    training_time = (datetime.now() - start_time).total_seconds()
                    
                    y_pred = model.predict(X_test)
                    
                    metrics = evaluate_regression(y_test, y_pred)
                    
                    # Cross-validation
                    cv_scores = cross_val_score(model, X, y, cv=5, scoring='r2')
                    metrics['cv_mean'] = safe_float(cv_scores.mean())
                    metrics['cv_std'] = safe_float(cv_scores.std())
                    
                    # Save predictions vs actual plot
                    import matplotlib.pyplot as plt
                    plt.figure(figsize=(8, 6))
                    plt.scatter(y_test[:100], y_pred[:100], alpha=0.5)
                    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
                    plt.xlabel('Actual')
                    plt.ylabel('Predicted')
                    plt.title(f'Predictions vs Actual - {name}')
                    pred_plot_path = REPORTS_DIR / f"{config.session_id}_{name}_pred.png"
                    plt.savefig(pred_plot_path)
                    plt.close()
                    
                    # Sanitize predictions for JSON (handle NaN)
                    pred_sample = [safe_float(x) for x in y_pred[:20].tolist()]
                    actual_sample = [safe_float(x) for x in y_test[:20].tolist()]
                    
                    result = {
                        'model_name': name,
                        'metrics': metrics,
                        'training_time': safe_float(training_time),
                        'parameters': model.get_params(),
                        'predictions_sample': pred_sample,
                        'actual_sample': actual_sample
                    }
                    results.append(result)
                    
                    # Best model based on R2 score
                    if metrics['r2_score'] > best_score:
                        best_score = metrics['r2_score']
                        best_model = {
                            'name': name,
                            'model': model,
                            'metrics': sanitize_for_json(metrics)
                        }
                
                except Exception as e:
                    results.append({
                        'model_name': name,
                        'error': str(e),
                        'metrics': {},
                        'training_time': 0,
                        'parameters': {}
                    })
    
    # Save best model
    if best_model:
        model_path = MODELS_DIR / f"{config.session_id}_best_model.pkl"
        with open(model_path, 'wb') as f:
            pickle.dump({
                'model': best_model['model'],
                'scaler': scaler,
                'label_encoders': label_encoders,
                'target_encoder': target_encoder,
                'features': features,
                'problem_type': problem_type
            }, f)
        
        session_data[config.session_id]['best_model'] = {
            'name': best_model['name'],
            'path': str(model_path),
            'metrics': sanitize_for_json(best_model['metrics'])
        }
    
    # Store results
    session_data[config.session_id]['training_results'] = results
    
    # Sanitize response to handle NaN values
    response_data = sanitize_for_json({
        'results': results,
        'best_model': {
            'name': best_model['name'] if best_model else None,
            'metrics': best_model['metrics'] if best_model else {}
        },
        'problem_type': problem_type
    })
    
    return JSONResponse(content=response_data)


@app.post("/select-features")
async def select_features(selection: FeatureSelection):
    """Select features and target for training"""
    if selection.session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    df = session_data[selection.session_id]['df']
    
    # Validate columns
    invalid_cols = [col for col in selection.features if col not in df.columns]
    if invalid_cols:
        raise HTTPException(status_code=400, detail=f"Invalid columns: {invalid_cols}")
    
    if selection.target and selection.target not in df.columns:
        raise HTTPException(status_code=400, detail=f"Invalid target column: {selection.target}")
    
    # Store selection
    session_data[selection.session_id]['features'] = selection.features
    session_data[selection.session_id]['target'] = selection.target
    session_data[selection.session_id]['problem_type'] = selection.problem_type
    
    # Auto-detect problem type if supervised
    if selection.problem_type != 'clustering' and selection.target:
        detected_type = detect_problem_type(df[selection.target])
        return {
            'message': 'Features selected successfully',
            'detected_problem_type': detected_type,
            'selected_problem_type': selection.problem_type
        }
    
    return {'message': 'Features selected successfully'}


@app.get("/models/{session_id}/download")
async def download_model(session_id: str):
    """Download the best trained model"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    if 'best_model' not in session_data[session_id]:
        raise HTTPException(status_code=404, detail="No model trained yet")
    
    model_path = session_data[session_id]['best_model']['path']
    return FileResponse(
        model_path, 
        media_type='application/octet-stream',
        filename=f"euron_best_model_{session_id[:8]}.pkl"
    )


@app.get("/models/available")
async def get_available_models():
    """Get list of all available models"""
    return {
        'classification': list(get_classification_models().keys()),
        'regression': list(get_regression_models().keys()),
        'clustering': list(get_clustering_models().keys())
    }


@app.get("/session/{session_id}/results")
async def get_session_results(session_id: str):
    """Get all results for a session"""
    if session_id not in session_data:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = session_data[session_id]
    
    return {
        'filename': session.get('filename'),
        'features': session.get('features', []),
        'target': session.get('target'),
        'problem_type': session.get('problem_type'),
        'training_results': session.get('training_results', []),
        'best_model': session.get('best_model', {})
    }


@app.delete("/session/{session_id}")
async def delete_session(session_id: str):
    """Delete a session and its data"""
    if session_id in session_data:
        # Clean up files
        session = session_data[session_id]
        if 'file_path' in session:
            try:
                os.remove(session['file_path'])
            except:
                pass
        if 'eda_report' in session:
            try:
                os.remove(session['eda_report'])
            except:
                pass
        if 'best_model' in session:
            try:
                os.remove(session['best_model']['path'])
            except:
                pass
        
        del session_data[session_id]
        return {'message': 'Session deleted successfully'}
    
    raise HTTPException(status_code=404, detail="Session not found")


# End of API endpoints


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
