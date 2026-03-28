# Climate & Health Advanced Ensemble Model

## Quick Start

### Prerequisites
- Python 3.7+
- pip package manager

### Setup (Choose One)

#### Option 1: Automatic Setup (Recommended)
```bash
cd Data
bash setup.sh
```

#### Option 2: Manual Install
```bash
cd Data
pip install -r requirements.txt
```

#### Option 3: Notebook Self-Install
The notebook cell #1 automatically installs all dependencies when run.

### Running the Model

```bash
cd Data
jupyter notebook Model/climate_health_advanced_model.ipynb
```

Then execute all cells in order from top to bottom.

## What the Model Does

### Architecture
- **Ensemble Strategy**: LightGBM + XGBoost + CatBoost + Random Forest
- **Feature Engineering**: 100+ engineered features including:
  - Temporal features (seasonal, cyclical encoding)
  - Climate stress indices (heat, drought, flood)
  - Age vulnerability interactions
  - Location aggregation statistics
  - Vegetation health indicators

- **Hyperparameter Tuning**: Optuna-based optimization
- **Stacking**: 10-fold stratified OOF (Out-of-Fold) blending
- **Threshold Optimization**: Custom threshold for F1/AUC trade-off

### Metric
Challenge Score = 0.60 × F1 + 0.40 × ROC-AUC (maximized with threshold optimization)

### Outputs
The notebook generates two submission files:
1. **submission_optimal_threshold.csv** - Uses optimized threshold
2. **submission_0.5_threshold.csv** - Uses standard 0.5 threshold

## Expected Performance
- **LGB OOF**: ~0.82-0.85
- **XGB OOF**: ~0.81-0.84
- **CatBoost OOF**: ~0.80-0.83
- **Ensemble OOF**: ~0.84-0.87 (before threshold optimization)
- **Target Leaderboard**: >0.84 (to beat baseline 0.839570)

## Data Files Required
Ensure these files are in the `Data/` directory:
- `Train.csv` - Training data (3,146 samples × 13 columns)
- `Test.csv` - Test data (1,030 samples × 12 columns)
- `climate_features.csv` - Climate aggregates (4,176 rows × 18 columns)

## Notebook Structure

| Cell | Section | Purpose |
|------|---------|---------|
| 1 | Dependencies | Auto-install required packages |
| 2 | Imports | Load all libraries |
| 3-4 | Data Loading | Load and merge train/test/climate data |
| 5 | Feature Engineering | Create 100+ engineered features |
| 6 | Location Features | Add location-based statistics |
| 7 | Prepare Features | Encode categoricals, handle missing values |
| 8-10 | Hyperparameter Tuning (Optuna) | Tune LightGBM, XGBoost, CatBoost |
| 11-14 | OOF Training | Train 10-fold cross-validation models |
| 15 | Ensemble Blending | Find optimal blend weights |
| 16 | Threshold Optimization | Optimize decision threshold for metric |
| 17 | Feature Importance | Visualize top 30 most important features |
| 18-19 | Generate Submission | Create submission CSVs |

## Key Hyperparameter Ranges

### LightGBM (Optuna optimizes over)
- Learning rate: 0.01 - 0.15
- Num leaves: 20 - 200
- Max depth: 3 - 12
- Feature fraction: 0.4 - 1.0
- Regularization (α, λ): 1e-8 - 10

### XGBoost
- Learning rate: 0.01 - 0.15
- Max depth: 3 - 10
- Subsample: 0.4 - 1.0
- Colsample bytree: 0.4 - 1.0
- Regularization (α, λ): 1e-8 - 10

### CatBoost
- Learning rate: 0.05 (fixed)
- Depth: 6 (fixed)
- L2 leaf reg: 3 (fixed)

## Troubleshooting

### Import Errors
```
ModuleNotFoundError: No module named 'lightgbm'
```
**Solution**: Run cell #1 to auto-install dependencies

### File Not Found Errors
```
FileNotFoundError: [Errno 2] No such file or directory: 'Train.csv'
```
**Solution**: Ensure you're running from the `Data/Model/` directory and the CSV files are in the `Data/` directory

### Memory Issues
If running on a machine with <4GB RAM:
- Reduce `N_FOLDS` from 10 to 5 in cell #2
- Reduce Optuna trial counts (lines ~400) from 30/20 to 10/10

### Slow Performance
- The hyperparameter tuning (Optuna section) takes ~10-30 minutes
- Full 10-fold OOF training takes ~15-40 minutes
- Total runtime: 45-90 minutes depending on system

## Advanced Customization

### Adjust Feature Engineering
Edit the `engineer_features()` function in cell #5 to add/remove features

### Change Ensemble Weights
Modify the blend weights in cell #15:
```python
w_lgb, w_xgb, w_cat = 0.30, 0.40, 0.30  # Custom weights
```

### Disable Optuna Tuning
Comment out cells #8-10 and use fixed best parameters

## Model Logic Flow

```
┌─────────────────────────────────────┐
│ 1. Load & Merge Data                │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ 2. Feature Engineering (100+ feats) │
└────────────┬────────────────────────┘
             │
┌────────────▼────────────────────────┐
│ 3. Prepare Features (Encode/Fill)   │
└────────────┬────────────────────────┘
             │
         ┌───┴────────────────────────┐
         │                            │
    ┌────▼────┐  ┌────────┐  ┌──────┐│
    │ Optuna  │  │ Optuna │  │ Fixed││
    │  LGB    │  │  XGB   │  │ CAT  ││
    └────┬────┘  └────┬───┘  └───┬──┘│
         │            │          │   │
    ┌────▼────┐  ┌────▼───┐  ┌──▼──┐│
    │10-Fold  │  │10-Fold │  │10-F ││
    │ LGB OOF │  │XGB OOF │  │ OOF ││
    └────┬────┘  └────┬───┘  └───┬──┘│
         │            │          │   │
         └────────────┼──────────┘   │
                      │              │
         ┌────────────▼──────────────┘
         │
    ┌────▼──────────────────┐
    │ Optimal Blend Weights │
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │ Threshold Optimization│
    └────┬───────────────────┘
         │
    ┌────▼──────────────────┐
    │   Generate Submission │
    └──────────────────────┘
```

## Files Generated
- `submission_optimal_threshold.csv` - Primary submission
- `submission_0.5_threshold.csv` - Fallback submission
- Visualizations in notebook:
  - Threshold optimization curve
  - Top 30 feature importances

## Contact & Tips
- For best results, ensure all training data is properly loaded
- The model is calibrated for the metric: 0.60×F1 + 0.40×AUC
- Consider ensemble weights for different metrics
