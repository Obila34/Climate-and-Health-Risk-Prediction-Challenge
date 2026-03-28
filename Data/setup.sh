#!/bin/bash

# Setup script for Climate & Health Advanced Model
# Run from the Data/ directory

echo "================================"
echo "Climate & Health Model Setup"
echo "================================"
echo ""

echo "[1/3] Installing Python dependencies..."
pip install -q pandas numpy scikit-learn lightgbm xgboost catboost optuna matplotlib seaborn jupyter

if [ $? -ne 0 ]; then
    echo "⚠ Some packages may not have installed successfully"
    echo "   You can manually install them or run the notebook which has auto-install"
else
    echo "✓ Dependencies installed"
fi

echo ""
echo "[2/3] Verifying data files..."
if [ -f "Train.csv" ] && [ -f "Test.csv" ] && [ -f "climate_features.csv" ]; then
    echo "✓ All data files present:"
    echo "  - Train.csv ($(wc -l < Train.csv) rows)"
    echo "  - Test.csv ($(wc -l < Test.csv) rows)"
    echo "  - climate_features.csv ($(wc -l < climate_features.csv) rows)"
else
    echo "✗ Some data files missing. Expected:"
    echo "  - Train.csv"
    echo "  - Test.csv"
    echo "  - climate_features.csv"
fi

echo ""
echo "[3/3] Ready to run!"
echo ""
echo "To run the notebook:"
echo "  jupyter notebook Model/climate_health_advanced_model.ipynb"
echo ""
echo "Or from command line:"
echo "  python -m jupyter notebook Model/climate_health_advanced_model.ipynb"
echo ""
