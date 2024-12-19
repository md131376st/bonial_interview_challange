from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GridSearchCV


def train_and_evaluate(x_train, y_train, x_val, y_val, models):
    best_model = None
    best_params = None
    best_score = 0
    best_model_name = None

    for model_name, model_info in models.items():
        print(f"Training {model_name}...")
        grid_search = GridSearchCV(
            estimator=model_info["model"],
            param_grid=model_info["params"],
            scoring="roc_auc",
            cv=3,  # Cross-validation for hyperparameter tuning
            verbose=1,
            n_jobs=-1
        )
        grid_search.fit(x_train, y_train)

        # Evaluate on validation set
        y_pred_proba = grid_search.best_estimator_.predict_proba(x_val)[:, 1]
        score = roc_auc_score(y_val, y_pred_proba)
        print(f"{model_name} ROC-AUC on Validation: {score:.4f}")

        # Save the best model
        if score > best_score:
            best_model = grid_search.best_estimator_
            best_params = grid_search.best_params_
            best_score = score
            best_model_name = model_name

    print(f"\nBest Model: {best_model_name}")
    print(f"Best Params: {best_params}")
    print(f"Best ROC-AUC: {best_score:.4f}")

    return best_model, best_model_name, best_params, best_score