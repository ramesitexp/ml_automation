# 🤖 EURON ML Automation

<div align="center">

![ML Automation](https://img.shields.io/badge/EURON-ML%20Automation-667eea?style=for-the-badge&logo=robot&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.9+-3776ab?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.109-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-ff4b4b?style=for-the-badge&logo=streamlit&logoColor=white)

**Automated Machine Learning System by me**

*Upload • Analyze • Train • Deploy*

</div>

---

## 📋 Overview

 ML Automation is a comprehensive, end-to-end automated machine learning platform that simplifies the entire ML workflow. From data upload to model deployment, this system handles everything automatically while providing detailed insights and comparisons.

### ✨ Key Features

- **📤 Data Upload**: Support for CSV, Excel, and JSON files
- **📊 Auto EDA**: Comprehensive exploratory data analysis using YData Profiling
- **🎯 Smart Feature Selection**: Automatic problem type detection
- **🤖 Multi-Model Training**: Train and compare multiple ML algorithms simultaneously
- **📈 Detailed Metrics**: In-depth performance metrics and visualizations
- **🏆 Best Model Selection**: Automatic identification of the best performing model
- **💾 Model Export**: Download trained models for deployment

---

## 🚀 Quick Start

### Prerequisites

- Python 3.9 or higher
- pip (Python package manager)

### Installation

1. **Clone or navigate to the project directory**
```bash
cd ml_automation
```

2. **Create and activate virtual environment**
```bash
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Start the application**
```bash
chmod +x run.sh
./run.sh
```

Or manually:
```bash
# Terminal 1 - Start Backend
cd backend
uvicorn main:app --host 0.0.0.0 --port 8000 --reload

# Terminal 2 - Start Frontend
cd frontend
streamlit run app.py --server.port 8501
```

5. **Access the application**
- **Frontend**: http://localhost:8501
- **API Docs**: http://localhost:8000/docs

---

## 📚 Supported Algorithms

### Classification
| Algorithm | Description |
|-----------|-------------|
| Logistic Regression | Linear model for binary/multiclass classification |
| SVM Classifier | Support Vector Machine with RBF kernel |
| Decision Tree Classifier | Tree-based classification |
| Random Forest Classifier | Ensemble of decision trees |
| KNN Classifier | K-Nearest Neighbors |
| Gradient Boosting Classifier | Sequential ensemble method |
| XGBoost Classifier | Extreme Gradient Boosting |

### Regression
| Algorithm | Description |
|-----------|-------------|
| Linear Regression | Ordinary least squares regression |
| Ridge Regression | L2 regularized linear regression |
| Lasso Regression | L1 regularized linear regression |
| SVR | Support Vector Regression |
| Decision Tree Regressor | Tree-based regression |
| Random Forest Regressor | Ensemble regression |
| KNN Regressor | K-Nearest Neighbors for regression |
| Gradient Boosting Regressor | Sequential ensemble for regression |
| XGBoost Regressor | Extreme Gradient Boosting for regression |

### Clustering
| Algorithm | Description |
|-----------|-------------|
| KMeans | Centroid-based clustering |
| DBSCAN | Density-based clustering |
| Agglomerative Clustering | Hierarchical clustering |

---

## 📊 Metrics & Evaluation

### Classification Metrics
- **Accuracy**: Overall correctness
- **Precision**: Positive predictive value
- **Recall**: True positive rate
- **F1-Score**: Harmonic mean of precision and recall
- **Confusion Matrix**: Visual representation of predictions
- **Cross-Validation Score**: 5-fold CV for robust evaluation

### Regression Metrics
- **R² Score**: Coefficient of determination
- **MSE**: Mean Squared Error
- **RMSE**: Root Mean Squared Error
- **MAE**: Mean Absolute Error
- **Cross-Validation Score**: 5-fold CV

### Clustering Metrics
- **Silhouette Score**: Cluster cohesion and separation
- **Davies-Bouldin Score**: Cluster similarity measure
- **Calinski-Harabasz Score**: Variance ratio criterion

---

## 🏗️ Architecture

```
ml_automation/
├── backend/
│   └── main.py              # FastAPI backend with ML logic
├── frontend/
│   └── app.py               # Streamlit frontend UI
├── data/                    # Uploaded datasets
├── models/                  # Trained model files
├── reports/                 # Generated EDA reports
├── requirements.txt         # Python dependencies
├── run.sh                   # Startup script
└── README.md               # This file
```

### Technology Stack

- **Backend**: FastAPI (Python web framework)
- **Frontend**: Streamlit (Python UI framework)
- **ML Framework**: Scikit-learn, XGBoost
- **EDA**: YData Profiling (Pandas Profiling)
- **Visualization**: Plotly, Matplotlib, Seaborn
- **Data Processing**: Pandas, NumPy

---

## 🔌 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/upload` | POST | Upload dataset file |
| `/data/{session_id}/preview` | GET | Preview uploaded data |
| `/data/{session_id}/statistics` | GET | Get statistical summary |
| `/data/{session_id}/eda-report` | POST | Generate EDA report |
| `/data/{session_id}/eda-report` | GET | Retrieve EDA report |
| `/select-features` | POST | Select features and target |
| `/train` | POST | Train ML models |
| `/models/available` | GET | List available models |
| `/models/{session_id}/download` | GET | Download trained model |
| `/session/{session_id}/results` | GET | Get training results |
| `/session/{session_id}` | DELETE | Delete session |

---

## 📱 User Interface

### 1. Upload Data
Upload your dataset in CSV, Excel, or JSON format. The system automatically detects column types and provides an overview.

### 2. Data Overview
- View data preview (head/tail)
- Statistical summaries
- Missing value analysis
- Correlation matrix

### 3. EDA Report
Generate comprehensive exploratory data analysis report with:
- Variable distributions
- Correlation analysis
- Missing value patterns
- Data quality alerts

### 4. Feature Selection
- Select feature columns (X)
- Select target column (y)
- Choose problem type (classification/regression/clustering)
- Automatic problem type detection

### 5. Model Training
- Configure training parameters
- Select algorithms to train
- Automatic preprocessing (scaling, encoding, imputation)
- Cross-validation

### 6. Results & Comparison
- Performance metrics for all models
- Comparison charts
- Confusion matrices (classification)
- Predictions vs Actual plots (regression)
- Best model identification

### 7. Download Model
Export the best trained model as a pickle file for deployment.

---

## 💡 Usage Example

```python
import pickle
import pandas as pd

# Load the trained model
with open('best_model.pkl', 'rb') as f:
    model_data = pickle.load(f)

# Extract components
model = model_data['model']
scaler = model_data['scaler']
label_encoders = model_data['label_encoders']
features = model_data['features']

# Prepare new data
new_data = pd.read_csv('new_data.csv')
X_new = new_data[features]

# Apply preprocessing
if scaler:
    numeric_cols = X_new.select_dtypes(include=['number']).columns
    X_new[numeric_cols] = scaler.transform(X_new[numeric_cols])

# Make predictions
predictions = model.predict(X_new)
print(predictions)
```

---

## 🛠️ Configuration

### Training Parameters
- **Test Size**: Proportion of data for testing (default: 0.2)
- **Random State**: Seed for reproducibility (default: 42)
- **Scale Features**: Standardize features (default: True)
- **Handle Missing**: Strategy for missing values (mean/median/mode/drop)

### Clustering Parameters
- **Number of Clusters**: For KMeans and Agglomerative (default: 3)
- **DBSCAN eps**: Neighborhood radius (default: 0.5)
- **DBSCAN min_samples**: Minimum points per cluster (default: 5)

---

## 🔒 Data Privacy

- All uploaded data is stored temporarily per session
- Data is automatically deleted when session ends
- No data is shared or stored permanently
- Local processing only

---

## 📝 License

Copyright © 2024  (Engagesphere Technology Private Limited)

All rights reserved.

---

## 🤝 Support

For support and inquiries:
- **Company**:  (Engagesphere Technology Private Limited)
- **Location**: Bengaluru, India
- **Website**: [euron.one](https://euron.one)

---

<div align="center">

**Built with ❤️ by me**

*Empowering businesses with intelligent automation*

</div>
