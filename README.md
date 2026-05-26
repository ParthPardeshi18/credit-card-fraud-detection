# Credit Card Fraud Detection 

## Problem statement 

The problem statement chosen for this project is to predict fraudulent credit card transactions with the help of machine learning models.

In this project, we will analyse customer-level data which has been collected and analysed during a research collaboration of Worldline and the Machine Learning Group. 

The dataset is taken from the [Kaggle Website](https://www.kaggle.com/mlg-ulb/creditcardfraud) website and it has a total of 2,84,807 transactions, out of which 492 are fraudulent. Since the dataset is highly imbalanced, so it needs to be handled before model building.

## Business Problem Overview

For many banks, retaining high profitable customers is the number one business goal. Banking fraud, however, poses a significant threat to this goal for different banks. In terms of substantial financial losses, trust and credibility, this is a concerning issue to both banks and customers alike.

It has been estimated by [Nilson report](https://nilsonreport.com/upload/content_promo/The_Nilson_Report_Issue_1164.pdf) that by 2020 the banking frauds would account to $30 billion worldwide. With the rise in digital payment channels, the number of fraudulent transactions is also increasing with new and different ways. 

In the banking industry, credit card fraud detection using machine learning is not just a trend but a necessity for them to put proactive monitoring and fraud prevention mechanisms in place. Machine learning is helping these institutions to reduce time-consuming manual reviews, costly chargebacks and fees, and denials of legitimate transactions.

## Understanding and Defining Fraud

Credit card fraud is any dishonest act and behaviour to obtain information without the proper authorization from the account holder for financial gain. Among different ways of frauds, Skimming is the most common one, which is the way of duplicating of information located on the magnetic strip of the card.  Apart from this, the other ways are:

- Manipulation/alteration of genuine cards
- Creation of counterfeit cards
- Stolen/lost credit cards
- Fraudulent telemarketing 

## Data Dictionary

The dataset can be download using this [link](https://www.kaggle.com/mlg-ulb/creditcardfraud)

The data set includes credit card transactions made by European cardholders over a period of two days in September 2013. Out of a total of 2,84,807 transactions, 492 were fraudulent. This data set is highly unbalanced, with the positive class (frauds) accounting for 0.172% of the total transactions. The data set has also been modified with Principal Component Analysis (PCA) to maintain confidentiality. Apart from ‘time’ and ‘amount’, all the other features (V1, V2, V3, up to V28) are the principal components obtained using PCA. The feature 'time' contains the seconds elapsed between the first transaction in the data set and the subsequent transactions. The feature 'amount' is the transaction amount. The feature 'class' represents class labelling, and it takes the value 1 in cases of fraud and 0 in others.


## Project Pipeline

The project pipeline can be briefly summarized in the following four steps:

- **Data Understanding:** Here, we need to load the data and understand the features present in it. This would help us choose the features that we will need for your final model.
- **Exploratory data analytics (EDA):** Normally, in this step, we need to perform univariate and bivariate analyses of the data, followed by feature transformations, if necessary. For the current data set, because Gaussian variables are used, we do not need to perform Z-scaling. However, you can check if there is any skewness in the data and try to mitigate it, as it might cause problems during the model-building phase.
- **Train/Test Split:** Now we are familiar with the train/test split, which we can perform in order to check the performance of our models with unseen data. Here, for validation, we can use the k-fold cross-validation method. We need to choose an appropriate k value so that the minority class is correctly represented in the test folds.
- **Model-Building/Hyperparameter Tuning:** This is the final step at which we can try different models and fine-tune their hyperparameters until we get the desired level of performance on the given dataset. We should try and see if we get a better model by the various sampling techniques.
- **Model Evaluation:** We need to evaluate the models using appropriate evaluation metrics. Note that since the data is imbalanced it is is more important to identify which are fraudulent transactions accurately than the non-fraudulent. We need to choose an appropriate evaluation metric which reflects this business goal.

## Folder structure

```
fraud-detection/
├── data/
│   └── creditcard.csv
├── notebooks/
│   ├── fraud_detection_pipeline.ipynb   ← main notebook
│   └── credit_card_fraud_detection.ipynb ← original group notebook
├── reports/
│   ├── SBM_CWK_Instructions_BUSM131.pdf
│   └── Solution-Approach.pdf
├── src/
│   ├── fraud_detection_pipeline.py
│   ├── _build_notebook.py
│   ├── data_pipeline.py                 ← shared splits (Ext.)
│   ├── setup_model.py                   ← retrains + persists XGBoost (Ext.)
│   ├── shap_analysis.py                 ← Extension 1
│   ├── calibration_analysis.py          ← Extension 2
│   └── fraud_app.py                     ← Extension 3 (Streamlit)
├── outputs/
│   ├── charts/                          ← output charts
│   ├── models/                          ← saved .pkl model files
│   ├── xgb_fraud_model.pkl              ← trained pipeline (Ext.)
│   ├── xgb_fraud_model_calibrated.pkl   ← isotonic-calibrated (Ext. 2)
│   ├── scaler.pkl                       ← fitted StandardScaler
│   ├── shap_*.png                       ← SHAP charts (Ext. 1)
│   ├── calibration_*.png                ← calibration charts (Ext. 2)
│   ├── threshold_optimisation.png       ← cost-sensitive sweep (Ext. 2)
│   └── model_comparison.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## Data

The dataset is not included in this repo due to file size (147MB).

Download `creditcard.csv` from:
[kaggle.com/datasets/mlg-ulb/creditcardfraud](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

and place it in the `/data` folder before running any notebooks.

## How to run

1. `pip install -r requirements.txt`
2. Place `creditcard.csv` in `data/`
3. Open `notebooks/fraud_detection_pipeline.ipynb`
4. Run the Streamlit app with: `streamlit run src/fraud_app.py`

## Personal extensions (beyond group submission)

Three follow-up artefacts built on top of the group submission, each
addressing a limitation flagged in the original report.

### Extension 1 — SHAP explainability  *(`src/shap_analysis.py`)*

Closes the report's gap: *"A separate explainability tool (SHAP) would be
needed if a customer or regulator asks for reasons."*

Five charts under `outputs/`:

- `shap_summary_beeswarm.png` — global feature impact distribution
- `shap_feature_importance.png` — top-15 features by mean |SHAP|
- `shap_waterfall_fraud.png` — most-confident TRUE FRAUD explanation
- `shap_waterfall_fp.png` — worst FALSE POSITIVE explanation
- `shap_dependence_v14.png` — SHAP vs raw V14, coloured by Amount

**Key finding:** V14, V10 and V4 are the top three drivers; the top-5
features explain ~52% of total |SHAP| magnitude. In the sampled test
data, the simple rule `V14 < -3` already captures a fraud rate of
~97% (vs the 0.17% base rate) — a useful sanity-check companion to
the full model.

### Extension 2 — Probability calibration  *(`src/calibration_analysis.py`)*

Closes the report's gap: *"Wrap the best estimator in
CalibratedClassifierCV so the output is a true probability."*

- Wraps the XGBoost pipeline in `CalibratedClassifierCV(method='isotonic', cv=5)`
- Brier score improves **0.000674 → 0.000606** (~10% reduction)
- Cost-sensitive threshold sweep reveals the **£-optimal threshold (~0.26)**
  differs from the **F1-optimal threshold (~0.65)** by ~39 points
- The £-optimal operating point recovers **£166 more net** than F1-optimal

Charts: `calibration_curve.png`, `calibration_distribution.png`,
`threshold_optimisation.png` (4-panel cost sweep).

### Extension 3 — Streamlit fraud risk scorer  *(`src/fraud_app.py`)*

A live, interactive demo. Run:

```bash
streamlit run src/fraud_app.py
```

Three tabs:

1. **Transaction risk scorer** — enter Amount, Time, V4/V10/V11/V12/V14/V17;
   get a real-time fraud probability, risk tier (AUTO-APPROVE / MANUAL REVIEW
   / AUTO-BLOCK), and a per-row SHAP explanation (top-5 feature contributions).
2. **Model performance summary** — KPI cards, ROC/PR curves, confusion
   matrix at the live threshold, model comparison table, business £ impact.
3. **Global SHAP explainability** — embeds the Extension 1 charts with
   plain-English captions for a risk committee.

Sidebar threshold slider updates the risk-tier boundaries in real time.

### Reproducing the extensions

```bash
python src/setup_model.py          # retrains + saves outputs/xgb_fraud_model.pkl
python src/shap_analysis.py        # Extension 1 charts
python src/calibration_analysis.py # Extension 2 charts
streamlit run src/fraud_app.py     # Extension 3 live app
```
