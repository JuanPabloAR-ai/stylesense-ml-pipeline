# Fashion Forward Forecasting — Review Recommendation Classifier

**Objective.** Build a clean, modular *machine learning pipeline* that predicts whether a customer recommends a product using text reviews, numeric, and categorical features.  
This project satisfies Udacity's rubric requirements: proper documentation, modular code, pipeline-based preprocessing, NLP features, hyperparameter tuning, and rigorous evaluation.

## Repository Structure
```
stylesense-ml-pipeline/
├─ StyleSense_Pipeline.ipynb
├─ src/
│  ├─ __init__.py
│  ├─ text_preproc.py
│  ├─ data_utils.py
│  └─ metrics.py
├─ data/                 # place the CSV here (not tracked by git)
│  └─ .gitkeep
├─ models/               # saved models (not tracked by git)
│  └─ .gitkeep
├─ requirements.txt
├─ .gitignore
└─ README.md
```

## Dataset
Use the women's clothing reviews dataset provided by the course (typical filename:  
`Womens Clothing E-Commerce Reviews.csv`). Put it under `data/` and **do not** commit it to git.

### Expected columns
- Target: `Recommended IND` (0/1)
- Text: `Review Text`, `Title`
- Numeric: `Age`, `Rating`, `Positive Feedback Count`
- Categorical: `Division Name`, `Department Name`, `Class Name`

> You can change column names directly in the notebook's **Config** section.

## Environment Setup
```bash
# (Optional) create a virtualenv
python -m venv .venv
# Windows: .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -r requirements.txt

# Download spaCy English model (once)
python -m spacy download en_core_web_sm
```

## How to Run
1. Place the CSV file in `data/`.
2. Open and run: `notebooks/StyleSense_Pipeline.ipynb` (run cells top to bottom).
3. The best trained pipeline will be saved to `models/model_pipeline.pkl`.

## Model Design (high level)
- **Preprocessing (ColumnTransformer):**
  - Numeric → median imputation + StandardScaler
  - Categorical → most-frequent imputation + OneHotEncoder
  - Text → spaCy lemmatization + TF‑IDF
- **Estimator:** Logistic Regression (you can swap in LinearSVC, RandomForest, etc.).
- **Tuning:** `GridSearchCV` with `StratifiedKFold`, optimizing F1.
- **Evaluation:** accuracy, precision, recall, F1, confusion matrix, ROC‑AUC (if available).

## Rubric Alignment
- **Code Quality:** modular, documented, PEP 8‑style names; clear notebook sections.
- **Model Pipeline:** end‑to‑end pipeline that handles numeric, categorical, and text features.
- **Machine Learning Model:** uses NLP preprocessing; performs hyperparameter tuning; evaluates on a held‑out test set with appropriate metrics.

## Reproducibility Notes
- Random seeds are fixed where appropriate.
- All preprocessing is inside the pipeline (no data leakage).
- Saved model can be loaded with `joblib.load("models/model_pipeline.pkl")`.

## Create a New GitHub Repository
1. Create an empty repo on GitHub (e.g., `stylesense-ml-pipeline`).
2. From your local folder containing these files:
```bash
git init
git branch -M main
git add .
git commit -m "Initial commit: StyleSense ML pipeline"
git remote add origin https://github.com/<YOUR_USERNAME>/stylesense-ml-pipeline.git
git push -u origin main
```
**Important:** Do not commit the dataset; `.gitignore` excludes `data/*` and `models/*` by default.

## License
Educational use. Add a license of your choice if you plan to distribute.
