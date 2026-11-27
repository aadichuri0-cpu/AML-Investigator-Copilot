
from sklearn.linear_model import LogisticRegression
from sklearn.base import clone
from sklearn.pipeline import Pipeline

from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

import pandas as pd

from backend.app.ml.preprocessing import preprocessor

from backend.app.ml.prepare_data import prepare_data

def create_model(class_weight=None):

    model_pipeline = Pipeline(
        steps=[
            ("preprocessing", clone(preprocessor)),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight=class_weight,
                ),
            ),
        ]
    )

    return model_pipeline


def train_model():

    # Get the original training and testing datasets
    X_train, X_test, y_train, y_test = prepare_data()


    # Split the training data into sub-training and validation sets
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )


    # Create a fresh machine learning pipeline
    model_pipeline = create_model()

    # Train the model using the sub-training data
    model_pipeline.fit(
        X_subtrain,
        y_subtrain,
    )

    # Create a class-weighted model
    balanced_model = create_model(class_weight="balanced")

    # Train it on the same training data
    balanced_model.fit(X_subtrain, y_subtrain)

    # Get the names of the features after preprocessing
    feature_names = (
        model_pipeline
        .named_steps["preprocessing"]
        .get_feature_names_out()
    )

    # Get the coefficients learned by Logistic Regression
    coefficients = (
        model_pipeline
        .named_steps["classifier"]
        .coef_[0]
    )

    # Create a table of features and their coefficients
    coefficient_df = pd.DataFrame({
        "feature": feature_names,
        "coefficient": coefficients,
    })

    # Sort features by the absolute size of their coefficients
    coefficient_df["absolute_coefficient"] = (
        coefficient_df["coefficient"].abs()
    )

    coefficient_df = coefficient_df.sort_values(
        by="absolute_coefficient",
        ascending=False,
    )

    # Display the 20 strongest coefficients
    print("\nTop 20 features by coefficient magnitude:")
    print(
        coefficient_df[
            ["feature", "coefficient"]
        ].head(20)
    )

    
    # ==========================================
    # Evaluate the original baseline model
    # ==========================================

    y_val_pred = model_pipeline.predict(X_val)

    print("\nOriginal Model Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred))

    print("\nOriginal Model Classification Report:")
    print(classification_report(
        y_val,
        y_val_pred,
        digits=4,
    ))

    # ==========================================
    # Evaluate the class-weighted model
    # ==========================================

    y_val_pred_balanced = balanced_model.predict(X_val)

    # Get the probability of laundering for each validation transaction
    y_val_prob_balanced = balanced_model.predict_proba(X_val)[:, 1]


    # Test different decision thresholds
    for threshold in [0.3, 0.4, 0.5,0.6,0.7]:

        # Convert probabilities into True or False predictions
        y_val_pred_threshold = (
            y_val_prob_balanced >= threshold
        )

        print(f"\nThreshold: {threshold}")

        print("Confusion Matrix:")
        print(confusion_matrix(
            y_val,
            y_val_pred_threshold,
        ))

        print("Classification Report:")
        print(classification_report(
            y_val,
            y_val_pred_threshold,
            digits=4,
        ))


    print("\nBalanced Model Confusion Matrix:")
    print(confusion_matrix(y_val, y_val_pred_balanced))

    print("\nBalanced Model Classification Report:")
    print(classification_report(
        y_val,
        y_val_pred_balanced,
        digits=4,
    ))


    print("\nModel training completed successfully!")


    # ==========================================
    # FINAL MODEL TRAINING
    # ==========================================

    print("\nRetraining the selected model on all training data...")

    balanced_model.fit(X_train, y_train)

    # ==========================================
    # FINAL TEST EVALUATION
    # ==========================================

    # Get laundering probabilities for unseen test transactions
    y_test_prob = balanced_model.predict_proba(X_test)[:, 1]

    # Apply our selected threshold
    final_threshold = 0.4

    y_test_pred = y_test_prob >= final_threshold

    print("\nFinal Test Threshold:", final_threshold)

    print("\nFinal Test Confusion Matrix:")
    print(confusion_matrix(y_test, y_test_pred))

    print("\nFinal Test Classification Report:")
    print(classification_report(
        y_test,
        y_test_pred,
        digits=4,
    ))

        # SAVE TRAINED MODEL

    artifact_dir = Path(__file__).resolve().parent / "artifacts"
    artifact_dir.mkdir(parents=True, exist_ok=True)

    model_artifact = {
        "model": balanced_model,
        "threshold": 0.4
    }

    model_path = artifact_dir / "aml_model.joblib"

    joblib.dump(model_artifact, model_path)

    print(f"\nModel saved successfully at: {model_path}")


    return (
        X_subtrain,
        X_val,
        y_subtrain,
        y_val,
        X_test,
        y_test,
    )


if __name__ == "__main__":

    train_model()





