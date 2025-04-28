## Environment / setup

- [] linux / wsl
- [] git
- [] python (3.9? 3.10?)
- [] LaTeX

- **Setup & Foundation**

  - [] Make sure everyone is on the same page with the project goals and expectations.
  - [] Ensure eveyrone has the right enviorment, and knows how things work.

( From the doc file )

- **Data Exploration & Preprocessing**

  - [x] Load training data (`train_FD001.txt`, etc.).
  - [ ] Perform initial EDA: check NaNs, view sensor statistics, plot sensor trends for a few engines.
  - [ ] Identify and potentially remove constant/non-informative sensor columns.
  - [ ] Visualize sensor correlations.
  - [ ] Write basic data loading and cleaning script/functions.
  - [ ] Explore test data and RUL file structure (though RUL file won't be used for initial labeling).

- **Feature Engineering**

  - [ ] Implement functions to calculate rolling window statistics (mean, std) for sensors.
  - [ ] Experiment with different window sizes.
  - [ ] Apply PCA/t-SNE to visualize engine trajectories in lower dimensions.
  - [ ] Document feature engineering choices and rationale in the report draft.

- **Phase 1: Clustering & Labeling**

  - [ ] Select relevant sensor features for clustering (potentially exclude operational settings).
  - [ ] Apply KMeans clustering (experiment with k=5, potentially others).
  - [ ] Apply Agglomerative Clustering (experiment with linkage methods).
  - [ ] Visualize clusters using PCA/t-SNE.
  - [ ] **Crucial:** Manually inspect sensor trends corresponding to cluster assignments. Align clusters to the 5 degradation stages (Normal -> Failure). This may require iteration and adjustment.
  - [ ] Document the cluster interpretation process and justification for stage alignment.
  - [ ] Create the final multi-stage label column based on the validated clustering.
  - [ ] Save the dataset with the new custom degradation stage labels.

- **Phase 2: Classification Model**

  - [ ] Split data into training and validation sets (consider engine ID for splitting - GroupKFold or similar).
  - [ ] Train baseline classification models (e.g., Logistic Regression, Random Forest) using custom stage labels.
  - [ ] Evaluate baseline performance (Accuracy, F1, Precision, Recall per class, Confusion Matrix). Focus on performance for later stages (3, 4).
  - [ ] Implement imbalance handling (e.g., `class_weight='balanced'` or SMOTE). Retrain and evaluate.
  - [ ] Experiment with more advanced classifiers (XGBoost, LightGBM).
  - [ ] Perform hyperparameter tuning (e.g., GridSearchCV).
  - [ ] Analyze feature importance for the best classifier.
  - [ ] Write unit tests for classification data preparation/prediction functions.
  - [ ] Document classification methodology, results, and findings in the report.

- **Phase 3: Regression Model**

  - [ ] Engineer the target variable: For each data point, calculate cycles remaining until the _next_ stage transition (based on custom labels).
  - [ ] Handle the final stage (Stage 4): Decide how to label time remaining (e.g., 0 or use original RUL if needed only for this calculation, or cap based on observation).
  - [ ] Split data for regression training/validation.
  - [ ] Train baseline regression models (e.g., Linear Regression, Random Forest Regressor).
  - [ ] Evaluate using RMSE, MAE, R².
  - [ ] Experiment with other regressors (Ridge, SVR, XGBoost).
  - [ ] Perform hyperparameter tuning.
  - [ ] Write unit tests for regression label generation/prediction functions.
  - [ ] Document regression methodology, results, and findings in the report.

- **Phase 4: Risk Score & Decision Logic**

  - [ ] Extract stage probabilities from the chosen classifier (ensure it supports `predict_proba`).
  - [ ] Get time-to-next-stage prediction from the chosen regressor. (Focus on time-to-failure, potentially by summing predictions or training a separate model for direct time-to-stage-4).
  - [ ] Implement Risk Score calculation (e.g., `Prob(Stage 4) / (Time_Left + epsilon)` or `Prob(Stage 4) * (Max_Time - Time_Left)`).
  - [ ] Experiment with normalization methods (Min-Max, etc.).
  - [ ] Define and tune an alert threshold based on the normalized Risk Score (e.g., using Precision-Recall analysis on identifying true high-risk situations).
  - [ ] Visualize Risk Score trends over cycles for selected engines.
  - [ ] Write unit tests for risk score calculation and normalization.
  - [ ] Document risk score formulation, normalization, thresholding, and results in the report.

- **Testing, Integration & Finalization**

  - [ ] Refine unit tests based on developed functions.
  - [ ] Integrate the steps into a coherent pipeline script (Data -> Cluster -> Label -> Classify -> Regress -> Risk Score).
  - [ ] Perform final evaluation on a hold-out test set (if applicable based on data splits).
  - [ ] Review and refactor code for clarity, efficiency, and documentation (docstrings, comments).
  - [ ] Ensure README.md is comprehensive (setup instructions, project overview, how to run).

- **Report Writing (LaTeX)**
  - [] IEEEtran?
