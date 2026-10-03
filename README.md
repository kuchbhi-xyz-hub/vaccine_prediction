# vaccine_prediction
End-to-end notebook predicting H1N1 and seasonal flu vaccination probabilities for 26,707 survey respondents. Covers structured missing data, model comparison and calibration. Tuned gradient boosting, test ROC-AUC 0.876 / 0.862.
# Flu Shot Learning: Predicting H1N1 and Seasonal Flu Vaccination

Predicting the probability that a survey respondent received the H1N1 vaccine and the seasonal flu vaccine, using their background, opinions and health behaviours.

The data comes from the United States National 2009 H1N1 Flu Survey, provided by the National Center for Health Statistics. Each row is one respondent who was asked whether they received each vaccine, along with questions about themselves.

## Problem

This is a multilabel problem with two binary targets, `h1n1_vaccine` and `seasonal_vaccine`. A respondent may have received neither vaccine, one of them, or both, so the two outcomes are not mutually exclusive. The required output is a probability for each vaccine rather than a yes/no label.

## Data

| | |
|---|---|
| Respondents | 26,707 |
| Features | 35 (plus `respondent_id`) |
| H1N1 vaccinated | 21.2% |
| Seasonal vaccinated | 46.6% |
| Train / test split | 21,365 / 5,342 (stratified on both targets) |

Features cover behaviours, doctor recommendations, health status, opinions about flu risk and vaccine effectiveness, demographics, and anonymised region, industry and occupation codes.

Two files are used: `features.csv` and `labels.csv`, joined one-to-one on `respondent_id`.

## Approach

**Data quality.** Missing values are widespread but structured. Only 24.1% of respondents answered every question, so dropping incomplete rows was not viable. Employment industry and occupation are blank for everyone who is not employed, because the question does not apply to them. Several question blocks go missing together. Missingness is also associated with the targets: respondents with a missing doctor-recommendation answer were H1N1 vaccinated at 8.4%, against 22.4% for those who answered.

**Preprocessing.** Every feature is treated as categorical with "Missing" as its own category, then one-hot encoded with categories under 25 training respondents grouped together. This keeps the missingness signal visible and avoids imposing a numeric scale on the opinion questions, where the "Don't know" code does not behave like a midpoint. The encoder is fitted inside each pipeline, so cross-validation folds do not leak information.

**Validation.** 5-fold cross-validation on the training set, with identical folds for every model and both targets. The test set is untouched until the final model is chosen. Metrics are ROC-AUC (main), average precision, log loss and Brier score, since the task asks for probabilities rather than labels.

## Results

Cross-validation ROC-AUC (training set):

| Model | H1N1 | Seasonal |
|---|---|---|
| Baseline (class prior) | 0.5000 | 0.5000 |
| Logistic regression | 0.8613 | 0.8579 |
| Logistic regression (balanced weights) | 0.8611 | 0.8579 |
| Random forest | 0.8627 | 0.8566 |
| Gradient boosting | 0.8642 | 0.8602 |
| Logistic regression (tuned) | 0.8618 | 0.8585 |
| **Gradient boosting (tuned)** | **0.8669** | **0.8617** |

Held-out test set, final model (tuned histogram gradient boosting, one per target):

| Target | ROC-AUC | Average precision | Log loss | Brier |
|---|---|---|---|---|
| H1N1 | 0.8761 | 0.7001 | 0.3382 | 0.1041 |
| Seasonal | 0.8621 | 0.8422 | 0.4630 | 0.1505 |

Gradient boosting beat tuned logistic regression in all five folds for both targets, by 0.0051 (H1N1) and 0.0032 (seasonal) on average.

The predicted probabilities are well calibrated. The mean predicted probability matches the actual rate (0.214 against 0.212 for H1N1, 0.466 against 0.466 for seasonal), and the largest gap in any calibration bin is 0.030 and 0.045.

## What drives the predictions

The two vaccines turned out to be different prediction problems. Permutation importance on the test set (drop in ROC-AUC when a feature is shuffled):

- **H1N1:** doctor recommendation for H1N1 (0.0832), health insurance (0.0635), perceived H1N1 risk (0.0360), perceived vaccine effectiveness (0.0293)
- **Seasonal:** perceived seasonal risk (0.0703), seasonal doctor recommendation (0.0477), perceived seasonal vaccine effectiveness (0.0452), age group (0.0308)

Age matters for the seasonal vaccine but barely for H1N1 (0.0025). Removing race, sex and income costs about 0.001 ROC-AUC, so a model that avoids these attributes is a realistic option.

These are associations learned from a single survey, not causal effects.

## Running it

```
pip install numpy pandas matplotlib seaborn scikit-learn jupyter
```

Place `features.csv` and `labels.csv` where the loading cell expects them, adjusting the paths at the top of the notebook if needed, then run the notebook from top to bottom. Random seeds are fixed, so results reproduce exactly. The tuning cells take several minutes.

The notebook writes `predictions.csv` with one row per test respondent and a predicted probability for each vaccine.

## Repository

```
PRCP-1014_vaccine_prediction.ipynb   complete analysis, from loading through conclusions
predictions.csv                      final model's test-set probabilities
```

## Limitations

- Opinions were recorded in the same survey as vaccination status, possibly after respondents were vaccinated, so a model predicting vaccination in advance may perform worse.
- The data covers one season (2009-10) in one country and may not transfer elsewhere.
- Region, industry and occupation are anonymised codes, which limits interpretation.
- Fairness across demographic groups was not assessed.

## Source

U.S. Department of Health and Human Services, National Center for Health Statistics. The National 2009 H1N1 Flu Survey. Hyattsville, MD: Centers for Disease Control and Prevention, 2012. Competition hosted by [DrivenData](https://www.drivendata.org/competitions/66/flu-shot-learning/).
