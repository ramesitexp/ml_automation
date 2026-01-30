# Quick Start Guide - See Everything Working!

## 🚀 Step-by-Step to See DVC + MLflow in Action

### Step 1: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 2: Run Setup Script (One-time setup)
```bash
python setup_dvc_mlflow.py
```

This will:
- Verify all packages are installed
- Set up MLflow tracking directory
- Initialize DVC (optional - MLflow works without it)
- Create all necessary directories

### Step 3: Start Backend (Terminal 1)
```bash
python -m uvicorn backend.main:app --host 0.0.0.0 --port 8001 --reload
```

**Wait for**: `INFO: Application startup complete`

### Step 4: Start Frontend (Terminal 2)
```bash
streamlit run frontend/app.py --server.port 8502
```

**Wait for**: `You can now view your Streamlit app in your browser`

### Step 5: Open Dashboard
Open: **http://localhost:8502**

### Step 6: Complete Workflow (See Everything!)

#### A. Upload Data
1. Click **"Upload Data"** in sidebar
2. Upload `sample_data/loan_approval.csv`
3. ✅ See data overview with statistics

#### B. Select Features  
1. Click **"Feature Selection"**
2. Problem Type: **Classification**
3. Features: Select all except `loan_approved`
4. Target: **loan_approved**
5. Click **"Confirm Selection"**
6. ✅ See target distribution chart

#### C. Train Models (THIS CREATES MLFLOW EXPERIMENTS!)
1. Click **"Train Models"**
2. Select models: **Logistic Regression**, **Random Forest Classifier**
3. Click **"Train Models"** button
4. ⏳ Wait for training (30-60 seconds)
5. ✅ See training results with metrics

#### D. View MLflow Experiments (NEW!)
1. Click **"MLflow Experiments"** in sidebar
2. ✅ **YOU WILL SEE:**
   - All training runs listed
   - Model names, accuracy, training time
   - Click a run to see detailed metrics and parameters
   - All hyperparameters logged automatically

#### E. View Model Registry (NEW!)
1. Click **"Model Registry"** in sidebar  
2. ✅ **YOU WILL SEE:**
   - Registered models (automatically registered after training)
   - Model versions
   - Stages (None/Staging/Production/Archived)
   - Select a model to see all versions
   - Transition models between stages

#### F. View DVC Versioning (NEW!)
1. Click **"DVC Versioning"** in sidebar
2. ✅ **YOU WILL SEE:**
   - DVC configuration
   - Commands (now visible with dark background!)
   - Benefits of DVC integration

### Step 7: Verify Everything Works

#### Check Backend Status
```bash
curl http://localhost:8001/status
```

#### Check MLflow Endpoints
```bash
# Get experiments
curl http://localhost:8001/mlflow/experiments

# Get runs
curl http://localhost:8001/mlflow/runs

# Get models
curl http://localhost:8001/mlflow/models
```

#### Run Test Script
```bash
python test_integration.py
```

## ✅ What You Should See

### After Training Models:

1. **Results Page**:
   - Best model highlighted
   - All models with metrics
   - MLflow run IDs for each model

2. **MLflow Experiments Page**:
   - Table of all runs
   - Metrics comparison
   - Detailed view of each run

3. **Model Registry Page**:
   - Registered models list
   - Version management
   - Stage transitions

4. **DVC Versioning Page**:
   - Configuration info
   - Commands (visible code blocks)
   - Integration benefits

## 🎯 Expected Results

- ✅ All text visible (dark on light, light on dark for code)
- ✅ Dropdowns visible (white background, dark text)
- ✅ MLflow tracking working automatically
- ✅ Model registry showing registered models
- ✅ DVC information displayed
- ✅ No errors in UI

## 🔧 If Something Doesn't Work

### "Error fetching registered models"
→ **This is normal if no models trained yet!** Train models first.

### "MLflow not initialized"  
→ Train a model to initialize MLflow automatically.

### Code blocks not visible
→ Refresh browser. They now have dark background (#1F2937) with light text.

### Dropdowns not visible
→ Refresh browser. All fixed with white backgrounds.

## 📊 What Gets Tracked Automatically

Every time you train models:

1. **MLflow Creates**:
   - Experiment run for each model
   - Logs all hyperparameters
   - Logs all metrics
   - Saves confusion matrices
   - Registers best model

2. **DVC Tracks**:
   - Dataset versions
   - Model file versions
   - Pipeline dependencies

## 🎉 Success Indicators

You'll know everything is working when:

- ✅ Training completes successfully
- ✅ MLflow Experiments page shows your runs
- ✅ Model Registry shows registered models
- ✅ All text is clearly visible
- ✅ Dropdowns work properly
- ✅ Code blocks are readable

**Everything is now integrated and ready to use!** 🚀
