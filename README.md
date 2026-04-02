# Effect of Socio-Economic Factors on Academic Performance

## Description
This repository contains a machine learning pipeline designed to analyze and predict students' academic performance—specifically their **Grade Point Average (GPA)**—based on a variety of socio-economic factors, demographic backgrounds, and their personal attitudes toward learning English.

The project compares multiple advanced machine learning algorithms. Despite the `_clf_` (classifier) naming convention in the source code files, the scripts actually implement **Regression** models to predict the continuous GPA values. To ensure robust and unbiased results, the methodology involves evaluating the models across **200 distinct holdout iterations**, with hyperparameter tuning performed via **Bayesian Optimization** for each split.

---

## Dataset Information
The dataset (`ds_v1.xlsx` / `ds_v1.csv`) is built from survey responses collected from university students. 

* **Target Variable:** * `Your recent GPA (Grade Points Average)`: Treated as a continuous variable for regression tasks.
* **Features (Independent Variables):**
  * **Demographics & Socio-Economic Status:** Gender, Department, Grade Level, High School Type, Family Accommodation Unit, Mother's/Father's Educational Level, and Family Income Level.
  * **Language Learning Perceptions:** Survey questions (mostly Likert-scale) measuring perceptions of English language difficulty, expected time to fluency, beliefs about age/aptitude, and attitudes towards pronunciation and native speakers.

---

## Code Information
The repository consists of one primary data processing script and several modeling scripts:

* **Data Processing (`read_data.py`):** Reads the Excel file, identifies categorical columns, applies **Label Encoding** to transform strings into integers, applies **Min-Max Scaling** to normalize all features, and separates the features from the target variable.
  
* **Machine Learning Models:**
  Each script initializes a specific regressor, splits the data, applies optimization, and records predictions.
  * `bayesian_clf_model.py`: Bayesian Ridge Regressor
  * `cat_boost_clf_model.py`: CatBoost Regressor
  * `ext_clf_model.py`: Extra Trees Regressor
  * `gb_clf_model.py`: Gradient Boosting Regressor
  * `kt_boost_clf_model.py`: KTBoost Regressor
  * `mlp_clf_model.py`: Multi-Layer Perceptron (MLP) Regressor
  * `rf_clf_model.py`: Random Forest Regressor
  * `svr_clf_model.py`: Support Vector Regressor (SVR)

---

## Methodology
The experimental setup follows a strict validation pipeline:
1. **Preprocessing:** Categorical encoding and Min-Max scaling of inputs.
2. **Data Splitting:** For each of the **200 holdouts**, the data is split into Training, Validation, and Test sets (10% test size, and 11% validation size from the remaining training pool).
3. **Hyperparameter Tuning:** **Gaussian Process Minimization** (`gp_minimize` from `skopt`) is utilized. It searches for optimal parameters over 50 iterations (`n_calls=50`), aiming to minimize the Mean Absolute Error (MAE) on the validation set.
4. **Evaluation:** The optimized model predicts the test set. Actual targets and predicted values are saved as `.txt` files for each holdout step.

---

## Requirements
To run this project, you need **Python 3.8+** and the following dependencies:
* `numpy`
* `pandas`
* `scikit-learn`
* `scikit-optimize`
* `catboost`
* `KTBoost`
* `interpret`
* `openpyxl`

---

## Usage Instructions (Installation & Execution)

### Step 1: Clone the Repository
Download the repository to your local machine:
```bash
git clone https://github.com/ysngrmz/effect_of_socio-economic.git
cd effect_of_socio-economic
```

### Step 2: Set Up a Virtual Environment (Recommended)
Create an isolated environment to avoid dependency conflicts.
* **Windows:**
```bash
python -m venv venv
venv\Scripts\activate
```
* **macOS/Linux:**
```bash
python3 -m venv venv
source venv/bin/activate
```

### Step 3: Install Dependencies
Install all required Python libraries via pip:
```bash
pip install numpy pandas scikit-learn scikit-optimize catboost KTBoost interpret openpyxl
```

### Step 4: Configure Directory Structure
The scripts expect a specific folder structure to read data and save outputs. Ensure your repository looks like this:
```text
effect_of_socio-economic/
├── data/
│   └── ds_v1.xlsx                 # Your dataset must be here
├── scripts/
│   ├── read_data.py
│   ├── rf_clf_model.py
│   └── ... (other models)
└── holdout_results/               # Create this folder if it doesn't exist!
```
*(Create the results folder using: `mkdir holdout_results`)*

### Step 5: Update Hardcoded Paths (Crucial Step)
Currently, the Python scripts contain absolute paths pointing to a specific server (`/truba/home/kokumusdagdeler/...`). **You must change these to relative paths so they work on your machine.**

Open `read_data.py` and each model script you intend to run, and modify the path injections.
**Change from:**
```python
sys.path.append('/truba/home/kokumusdagdeler/yasin/effect_of_socio/scripts/')
inputs, targets, columns, col_val_list = rd.read_data(data_read_dir='/truba/home/kokumusdagdeler/yasin/effect_of_socio/data/')
```
**Change to (assuming you run scripts from the project root):**
```python
sys.path.append('./scripts/')
inputs, targets, columns, col_val_list = rd.read_data(data_read_dir='./data/')
```

### Step 6: Run a Model
Once paths are updated, execute any model script from your terminal:
```bash
python scripts/rf_clf_model.py
```
*Note: The code performs 200 holdout splits and 50 hyperparameter tuning steps per split. Running a single model script completely is computationally intensive and will take time.*

### Step 7: View Results
After execution, check the `holdout_results/` directory. You will find text files (e.g., `targets_for_holdout_1`, `preds_for_holdout_1`) containing the actual and predicted GPA arrays for performance evaluation.



