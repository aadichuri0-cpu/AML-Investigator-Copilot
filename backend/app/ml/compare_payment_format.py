
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

from backend.app.ml.prepare_data import prepare_data
from backend.app.ml.train_model import create_model
from backend.app.ml.preprocessing import (
    categorical_features,
    numerical_features,
    boolean_features,
)


def create_model_without_payment_format():

    # Remove only payment_format from the categorical features
    categorical_without_payment = [
        feature
        for feature in categorical_features
        if feature != "payment_format"
    ]

    # Create a separate preprocessing pipeline
    experimental_preprocessor = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                categorical_without_payment,
            ),
            (
                "numerical",
                StandardScaler(),
                numerical_features,
            ),
            (
                "boolean",
                "passthrough",
                boolean_features,
            ),
        ]
    )

    # Create the experimental model
    experimental_model = Pipeline(
        steps=[
            ("preprocessing", experimental_preprocessor),
            (
                "classifier",
                LogisticRegression(
                    max_iter=1000,
                    class_weight="balanced",
                ),
            ),
        ]
    )

    return experimental_model


def compare_models():

    # Load the original train/test split
    X_train, X_test, y_train, y_test = prepare_data()

    # Recreate the same validation split
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    threshold = 0.4

    # ==========================================
    # MODEL A: Current model with all features
    # ==========================================

    current_model = create_model(class_weight="balanced")

    current_model.fit(X_subtrain, y_subtrain)

    current_probabilities = current_model.predict_proba(X_val)[:, 1]

    current_predictions = current_probabilities >= threshold

    # ==========================================
    # MODEL B: Experimental model without payment_format
    # ==========================================

    experimental_model = create_model_without_payment_format()

    experimental_model.fit(X_subtrain, y_subtrain)

    experimental_probabilities = (
        experimental_model.predict_proba(X_val)[:, 1]
    )

    experimental_predictions = (
        experimental_probabilities >= threshold
    )

    # ==========================================
    # COMPARE BOTH MODELS
    # ==========================================

    print("\nMODEL A: Current Model")
    print("---------------------")

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val,
        current_predictions,
        labels=[False, True],
    ))

    print("Classification Report:")
    print(classification_report(
        y_val,
        current_predictions,
        digits=4,
    ))

    print("\nMODEL B: Without payment_format")
    print("--------------------------------")

    print("Confusion Matrix:")
    print(confusion_matrix(
        y_val,
        experimental_predictions,
        labels=[False, True],
    ))

    print("Classification Report:")
    print(classification_report(
        y_val,
        experimental_predictions,
        digits=4,
    ))


if __name__ == "__main__":
    compare_models()
