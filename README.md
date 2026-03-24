# Socio-Demographic Effects Prediction Models

This repository contains various machine learning regression models designed to analyze and predict target variables (e.g., GPA) based on socio-demographic survey data. The pipeline includes data preprocessing and automated hyperparameter tuning using Bayesian Optimization (`skopt`).

## Project Structure

* `data/`: Directory where the dataset (`ds_v1.xlsx`) should be stored.
* `scripts/read_data.py`: A utility script to load the data, handle string label encoding, and perform min-max scaling.
* **Model Scripts**: Python scripts for training and evaluating different regression algorithms. Each script uses Gaussian Process minimization (`gp_minimize` from `skopt`) for hyperparameter optimization and runs 200 holdout validation iterations.
    * `bayesian_clf_model.py`: Bayesian Ridge Regressor
    * `cat_boost_clf_model.py`: CatBoost Regressor
    * `ext_clf_model.py`: Extra Trees Regressor
    * `gb_clf_model.py`: Gradient Boosting Regressor
    * `kt_boost_clf_model.py`: KTBoost Regressor
    * `mlp_clf_model.py`: Multi-Layer Perceptron (Neural Network) Regressor
    * `rf_clf_model.py`: Random Forest Regressor
    * `svr_clf_model.py`: Support Vector Regressor (SVR)
* `holdout_results/`: Directory where the actual targets and model predictions for each holdout iteration are saved as text files.

## Prerequisites

To run these scripts, you will need Python installed along with the following packages:

* `numpy`
* `pandas`
* `scikit-learn`
* `scikit-optimize` (`skopt`)
* `catboost`
* `KTBoost`
* `openpyxl` (for reading Excel data files)

You can install the dependencies via pip:

```bash
pip install numpy pandas scikit-learn scikit-optimize catboost KTBoost openpyxl
