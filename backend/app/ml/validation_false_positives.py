
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

from backend.app.ml.prepare_data import prepare_data
from backend.app.ml.train_model import create_model


def investigate_validation_false_positives():

    # 1. Load the original train/test split
    X_train, X_test, y_train, y_test = prepare_data()

    # 2. Recreate the exact validation split from train_model.py
    X_subtrain, X_val, y_subtrain, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.2,
        random_state=42,
        stratify=y_train,
    )

    # 3. Create and train the balanced model
    model = create_model(class_weight="balanced")

    model.fit(X_subtrain, y_subtrain)

    # 4. Predict laundering probabilities on validation data
    probabilities = model.predict_proba(X_val)[:, 1]

    # 5. Apply the same threshold used in your project
    threshold = 0.4

    predictions = probabilities >= threshold

    # 6. Create a table for validation predictions
    validation_results = X_val.copy()

    validation_results["actual"] = y_val
    validation_results["probability"] = probabilities
    validation_results["predicted"] = predictions

    # 7. Identify false positives
    false_positives = validation_results[
        (validation_results["actual"] == False)
        & (validation_results["predicted"] == True)
    ]

    print("\nValidation Confusion Matrix:")
    print(confusion_matrix(y_val, predictions))

    print("\nTotal Validation Transactions:", len(X_val))
    print("Total False Positives:", len(false_positives))


    # Compare false-positive rates across transaction groups
    legitimate_transactions = validation_results[
        validation_results["actual"] == False
    ].copy()

    legitimate_transactions["transaction_group"] = (
        legitimate_transactions["payment_format"].eq("ACH")
        .map({True: "ACH", False: "Non-ACH"})
        + " + "
        + legitimate_transactions["is_cross_border"]
        .map({True: "Cross-border", False: "Domestic"})
    )

    group_analysis = legitimate_transactions.groupby(
        "transaction_group"
    )["predicted"].agg(
        legitimate_count="count",
        false_positive_count="sum",
        false_positive_rate="mean",
    )

    group_analysis["false_positive_rate"] = (
        group_analysis["false_positive_rate"] * 100
    )

    print("\nFalse-Positive Rate by Transaction Group:")
    print(group_analysis.round(2))


    print("\nFalse Positive Rate by Payment Format:")
    print(
        false_positives["payment_format"]
        .value_counts()
    )

    print("\nFalse Positives by Cross-Border Status:")
    print(
        false_positives["is_cross_border"]
        .value_counts()
    )

    print("\nFalse Positives: ACH + Cross-Border")
    ach_cross_border_fp = false_positives[
        (false_positives["payment_format"] == "ACH")
        & (false_positives["is_cross_border"] == True)
    ]

    print("Count:", len(ach_cross_border_fp))

    print("\nSample ACH + Cross-Border False Positives:")
    print(
        ach_cross_border_fp[
            [
                "payment_format",
                "is_cross_border",
                "amount_received",
                "from_country",
                "to_country",
                "probability",
            ]
        ].head(10)
    )


if __name__ == "__main__":
    investigate_validation_false_positives()
