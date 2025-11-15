# FDS_COMP_2025 — Kaggle Competition (Pokémon Battle Prediction 2025)

This repository contains three independent models designed to generate submissions for the Kaggle competition **“FDS — Pokémon Battles Prediction 2025.”**  
Each model is provided as a separate Python script.

## Repository Contents

- **model-1.py** — Model 1  
- **model-2.py** — Model 2  
- **model-3.py** — Model 3  

Each script implements a complete model and produces its own output file.
















###############################################################################################################################################################################
# FDS challenge: DEB | MRK | VIT
## SUBMISSION 1 - for *"fork-of-vit-notebook-fds2-final.py"*
- **Feature engineering**:
    -Comprehensive, deterministic pipeline that combines static team information, per‑pokemon base stats, timeline-derived signals, move details, matchup heuristics and global aggregated statistics.
    - Static team stats: team size, unique type counts, sums and averages of base stats (base_hp, base_atk, base_def, base_spa, base_spd, base_spe), team stat variance, and team speed summaries (mean, max). Also counts of how many team members out‑speed the opponent’s lead.
    - Timeline-derived HP features: for each battle the code extracts p1/p2 hp_pct series (up to 30 turns) and computes mean, last, std, min, slope, AUC, fraction of turns in advantage, KO counts, early-window summaries (first 3/5/10), and first‑KO flags.
Move-level statistics: per-side (p1/p2) move base_power, accuracy and priority means and maxima for full and early windows (5/10). Also counts per category (PHYSICAL/SPECIAL/STATUS).
    - Type effectiveness & STAB: per-move type multipliers using a TYPE_CHART; averages of effectiveness for P1 moves vs P2 lead (full/5/10), STAB ratios/power means for P1 and lead-only approximations for P2.
    - Matchup/damage proxies: a simple damage-index combining base_power, STAB, type multiplier and attack/defense proxies; a lead matchup index computed over full/5/10 windows.
    - Momentum, hazards and tactical signals: switch counts, hazard presence flags, recovery counts, momentum shift (3 vs 10 turns), damage_trade_ratio (weighted), forced/voluntary leave heuristics, late/early game differences and “final battle score” heuristics with a sigmoid-derived final_win_probability.
    - Global training-time stats (computed once from the whole train set and merged back): per-Pokémon winrates (Laplace-smoothed), per-Pokémon average final HP, per-Pokémon average damage. Those are aggregated to team-level features (team_winrate_score, team_avg_hp_score, avg_damage features).
    - Interaction & composite features: manual multiplications (team strength × move power, speed × priority, etc.), “extra” composite features such as damage × status, signed-log transforms for skewed features, and many bounded, safe fallbacks to avoid NaN/Inf.
    - Clustering-based meta feature: KMeans clustering over the scaled feature matrix (automatic best-K selection using silhouette), then cluster → cluster_win_rate mapping is computed and appended as cluster_id and cluster_win_rate features for train and test.
Sanitization & scaling: fillna/clip/replace inf steps; final RobustScaler fitted on train numeric columns and applied to both train and test. Dataframes train_df_raw / test_df_raw are kept as raw copies for inspection.
- **Feature selection**:

    - Correlation and constant pruning: a utility drops constant columns and then removes one feature from pairs with |corr| > 0.92 (this reduces redundancy before later selection).
    - L1-based selection: the notebook fits an L1 LogisticRegression on a train/val split (balanced class option used in one experiment), takes absolute coefficients and keeps features with coefficient magnitude above a small threshold (example: 0.003). This produces a selected feature set used for a stronger retrained L1 model.
    - Practical pipeline: the notebook shows both a correlation-pruned set (used for several experiments) and an L1-selected smaller subset (used for final compact Logistic models). It preserves the contract expected by downstream cells (variables like model, features, train_selected, test_selected).
- **Modeling & stacking**:
    - Base models explored and tuned inside the notebook:
        - Logistic Regression (baseline): standard solver, used as an initial baseline and retrained on various subsets.
        - L1 Logistic Regression: used for sparsity-aware feature selection and as a competitive base learner.
        - K-Nearest Neighbors: K selected by CV across candidate k values (example list: 3,5,7,...,31), then fitted and used as a base learner.
        - Random Forest: two profiles compared via CV ("shallow" vs "deep") and best chosen; used as a base learner and for feature importances in experiments.
        - XGBoost: hist tree method with sensible defaults (e.g., n_estimators=600, max_depth=5, learning_rate=0.05, subsample/colsample near 0.9). CV accuracy reported and then fit as a base learner.
        - *Stacking ensemble*: The notebook implements an OOF stacking (clean OOF) workflow using StratifiedKFold for out-of-fold predictions (the default stacking CV used in the stacking cell is the same CV object used earlier; common choice in the notebook is 5 folds). Base learners in the final stack are: L1 Logistic, RandomForest (best config), KNN (best k), and XGBoost. For each fold, base models are trained on the training folds, OOF probabilities for that fold are stored for each base learner, creating a stacking training matrix shaped (n_train, n_base_models). Meta-learner is a LogisticRegression on the stacked OOF probabilities (L2 regularization). The meta is trained on the concatenated OOF predictions. After OOF meta training, base learners are refit on the full train data and used to produce test-level base probabilities; those are stacked and fed to the meta to produce final test scores/labels.
- **Threshold tuning**:
    - The notebook performs an OOF-level threshold search for the stacking meta-model:
        - It re-fits the meta in a cross-validated OOF manner and gathers OOF meta probabilities.
        - It then scans thresholds in the 0.30–0.71 range (step 0.005). For each threshold it computes accuracy on the OOF labels (other metrics can be computed similarly).
        - The best threshold is chosen as the one maximizing OOF accuracy (the code prints the best threshold and OOF accuracy).
        - The chosen threshold is applied to the final stacked test scores (from the fully-trained meta) to produce stack_pred_labels_tuned and the submission binary labels.

## SUBMISSION 2 - for *"fork-of-vit-notebook-fds.py"*
- **Feature engineering**:
    - A large, deterministic FE pipeline that extracts static team stats, timeline-based HP series, move-based features (power/accuracy/priority), type effectiveness, STAB, early/mid/late momentum, hazards/switches, damage proxies, and many composite engineered features (signed log transforms, top-5 “extra strong” features, stack-oriented meta-features).
    - Global training-time stats: per-Pokémon winrates, average HP stats, average damage computed from full train set and used as features.
    - Robust sanitization: fillna/finite clipping, signed-log transforms, and a final RobustScaler applied to numeric columns.
    - Some final manual interactions and engineered meta-features (10 safe features + top-5 extra features + XGB-inspired transforms).
- **Feature selection**:
    - Constant-feature removal, correlation pruning for LR (|ρ|>0.95), then per-model Top-K (LR/XGB/RF) computed via LR coefficient magnitudes, XGB feature_importances_, RF feature_importances_. Union of selected features used to create reduced train/test frames.
- **Modeling & stacking**:
    - Base learners: LogisticRegression (scaled), XGBoost (hist tree, early stopping, then sigmoid calibration), RandomForest (calibrated). Default seeds and sensible hyper-parameter choices are present (e.g., XGB: n_estimators=2000, lr=0.03, max_depth=6; RF with n_estimators=400, max_depth=8).
    - K-fold OOF stacking: FOLDS = 20 stratified folds. For each fold, base learners are trained, their calibrated val probabilities stored in oof_base; per-fold test probabilities are stored and averaged.
    - Meta-learner: LogisticRegression on the 3-column OOF probability matrix. There is also a small grid search over meta C and random_state to pick best meta config (replaces oof_meta_scores / meta_test_scores with best config).
- **Threshold tuning**:
    - Scan thresholds between 0.30 and 0.70 (401 values) on meta OOF probabilities. Metrics considered: accuracy, F1, MCC, ROC-AUC, logloss. Selected operating threshold prioritizes accuracy > F1 > MCC ties.

## SUBMISSION 3 - for *"mrk-notebook-fds.py"*
- **Feature engineering**
    - High-level approach:
        - Extensive handcrafted features built from each battle's JSON record and its battle_timeline. Features include static team stats, timeline-derived HP/momentum signals, move statistics, type-effectiveness, hazards, switches, recoveries, and several domain heuristics.
    - Key helper functions and patterns:
        - Timeline and HP: get_timeline, _extract_hp_series, _mean_last_std_min, _slope, _auc_pct, _ko_count.
        - Move stats and windows: _move_stats_for_side, which derives mean base_power / accuracy / priority (both full and windowed).
        - Type & STAB: _type_multiplier, _avg_type_eff_p1_vs_p2lead, _name_to_types_map_p1, _stab_features.
        - Early-game and momentum: _early_momentum_features, _first_ko_flag, _priority_counts, _priority_feature_block.
        - Lead matchup / damage index: _simple_damage_index, _p1_vs_p2lead_matchup_index that computes simple damage proxies combining base_power, STAB, type effectiveness, and atk/def proxies.
        - Switch, hazard and recovery heuristics: _switch_count, _hazard_flags, _recovery_pressure, _momentum_shift.
        - Aggregation helpers: new_features, new_features_deb, new_features_mirko which produce many higher-level, human-interpretable features (e.g., final_hp_winner, team strength gap, predicted_win_prob, final_battle_score, final_win_probability).
        - Global statistics built from train set: build_pokemon_win_stats, build_pokemon_hp_stats, build_pokemon_avg_damage — used to create team-level win-rate / HP / avg-damage features.
        - Single-record assembly: _one_record_features collects static team metrics, timeline HP summaries, type/lead indices, move-derived metrics, and calls the new_features* blocks and damage features; create_simple_features applies _one_record_features to all records and returns a DataFrame.
    - Interaction & engineered features:
        - Post-processing creates additional interaction columns (_maybe_add_interactions) such as speed × priority, type-effectiveness × STAB, lead_matchup × early momentum, etc.
        - A separate “10 safe, high-signal features” block constructs robust aggregated features (e.g., atk_def_ratio, spd_gap, hp_ratio, survival_score, momentum_index, power_acc_gap, offensive_balance, defensive_efficiency, status_influence, speed_ratio). These are sanitized (clip, safe divide, float32) and validated for NaN/Inf.
    - Data sanitation:
        - Convert numeric columns to float32, replace +/-inf with NaN, clip percent-like fields (hp/auc) into sensible bounds, and detect near-constant columns (for informational purposes).
        - Utilities for safe division, pick-first from candidate columns, and normalization of accuracy features.
- **Feature selection**
        - Train-only pruning followed by a learned selector (Elastic Net logistic with CV). The pipeline purposefully avoids leaking test information and performs multiple pruning steps before the selector.
        - Pruning steps in order:
            - Constant removal: drop columns with ≤1 distinct observed value.
            - Correlation pruning: compute Pearson correlation on imputed train data; drop one side of pairs with |ρ| > CORR_THR (CORR_THR = 0.99).
            - Robust VIF pruning: iterative VIF computed on standardized, imputed train columns. Config: DROP_VIF True, VIF_THRESHOLD = 25.0, MAX_VIF_STEPS = 50. The VIF uses sklearn LinearRegression to compute R^2 per-column, dropping highest-VIF until below threshold or max steps.
        - Elastic Net selector:
            - Model: LogisticRegressionCV with penalty="elasticnet", solver="saga".
            - Post-fit selection: take absolute coefficients, then select a top-N by absolute weight. 
- **Modelling**
        - Overall architecture:
            - Base learners: Logistic Regression, XGBoost, Random Forest.
            - Stacking: true out-of-fold (OOF) stacking with per-fold training of base learners, calibration of probabilistic outputs on the validation fold, then a logistic meta-learner trained on base OOF probabilities.
            - Multi-seed runs: the notebook provides a multi-seed ensemble wrapper that repeats the stacking pipeline for multiple seed combinations and selects the best OOF AUC run.
        - Base learner details and hyperparameters:
            - Logistic Regression (base/layer-0): Pipeline: StandardScaler + LogisticRegression(solver="liblinear", penalty="l2", C=0.5, max_iter=3000). No calibration applied (probabilities from LR are used directly).
            - XGBoost:
                - XGBClassifier params: n_estimators=2000 (early stopping expected to cut it), learning_rate=0.03, max_depth=6, subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0, reg_alpha=0.0, objective="binary:logistic", eval_metric="logloss", tree_method="hist", random_state variable per run.
                - Trained with early stopping against the fold validation set; after fit a per-fold sigmoid calibration (CalibratedClassifierCV with cv="prefit") is applied on the validation fold.
            - Random Forest:
                - RandomForestClassifier(n_estimators=400, max_depth=10, min_samples_leaf=10, max_features="sqrt", bootstrap=True, n_jobs=-1).
                - Per-fold sigmoid calibration (CalibratedClassifierCV with cv="prefit") on the fold validation.
        - Stacking details:
            - K-folds: FOLDS = 10 (StratifiedKFold shuffle=True). For each fold: train base learners on fold train, predict probs on fold val (store in oof_base) and on test (store per-fold test predictions).
            - After folds: aggregate test probs by averaging across folds to produce test_base_mean.
            - Meta-learner: LogisticRegression(solver="lbfgs", penalty="l2", C=1.0, max_iter=5000). The notebook builds a true OOF for the meta via a second StratifiedKFold over the base OOF features, fits meta per meta-fold, and averages meta predictions on test across meta folds.
             - Multi-seed ensemble: the notebook runs several seed combinations (explicit tuple list) repeating the entire stacking pipeline and records each run’s OOF AUC; the best run is selected (best_result) for threshold tuning/submission.
        - Calibration:
            - Both XGBoost and RandomForest receive per-fold sigmoid calibration using CalibratedClassifierCV(prefit) fitted on the validation fold to improve probability quality and prevent calibration leakage.
- **Threshold tuning**
    - Choose the final binary decision threshold for the stacked meta OOF probabilities to maximize a chosen metric (accuracy in this notebook).
    - Procedure implemented:
        - Baseline accuracy computed at threshold 0.50: oof_acc_default = accuracy_score(y, (oof_meta_scores >= 0.50).astype(int)).
        - Coarse search: thresholds ths_coarse = np.linspace(0.30, 0.70, 121) (step ≈ 0.0033). Compute accuracy across those thresholds, pick coarse best.
        - Fine search: define a narrow window around the coarse best (±0.05), then search ths_fine = np.arange(fine_lo, fine_hi + 1e-12, 0.001) (step 0.001). Pick the threshold with the highest OOF accuracy.
        - Store final values: STACK_FINAL_THRESHOLD, STACK_FINAL_OOF_ACC. Apply threshold to meta_test_scores to produce stack_pred_labels_tuned for submission.
