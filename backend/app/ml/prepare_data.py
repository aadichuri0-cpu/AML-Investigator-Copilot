
from pathlib import Path

import pandas as pd

from sklearn.model_selection import train_test_split
from backend.app.ml.preprocessing import prepare_features

PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "aml_features.csv"
)


def load_processed_data():
    """
    Load the processed AML transaction dataset.
    """

    df = pd.read_csv(DATA_PATH)

    return df


def prepare_data():
    """
    Load the processed AML dataset,
    separate features and target,
    and split into training and testing data.
    """

    # Load the processed dataset
    transactions = load_processed_data()

    # Separate features and target
    excluded_columns = [
        "is_laundering",
        "predicted_alert",
        "model_score",
        "record_key",
        "from_account",
        "to_account",
        "is_dashboard_sample",
        "sender_transaction_count",
        "sender_unique_receivers",
        "sender_total_amount",
        "timestamp",
    ]

    X = transactions.drop(columns=excluded_columns)
    y = transactions["is_laundering"]

    # Split the dataset
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Return the original features and labels
    return X_train, X_test, y_train, y_test



if __name__ == "__main__":

    # Load the processed dataset
    transactions = load_processed_data()

    # Convert timestamp text into actual datetime values
    transactions["timestamp"] = pd.to_datetime(
        transactions["timestamp"]
    )

    # Sort transactions from oldest to newest
    transactions = transactions.sort_values("timestamp")

    print("Dataset shape:")
    print(transactions.shape)

    print("\nFirst 5 transactions:")
    print(transactions.head())

    print("\nFull target distribution:")
    print(transactions["is_laundering"].value_counts())

    # Columns that must not be used as model features
    excluded_columns = [
        "is_laundering",
        "predicted_alert",
        "model_score",
        "record_key",
        "from_account",
        "to_account",
        "is_dashboard_sample",
        "sender_transaction_count",
        "sender_unique_receivers",
        "sender_total_amount",
        "timestamp",
    ]

    # Separate features and target
    X = transactions.drop(columns=excluded_columns)

    y = transactions["is_laundering"]

    # Split the dataset into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y,
    )

    # Apply preprocessing to training and testing features

    X_train_processed, X_test_processed = prepare_features(
        X_train,
        X_test,
    )

    print("\nProcessed training data shape:")
    print(X_train_processed.shape)

    print("\nProcessed testing data shape:")
    print(X_test_processed.shape)

    print("\nTraining features shape:")
    print(X_train.shape)

    print("\nTesting features shape:")
    print(X_test.shape)

    print("\nTraining target distribution:")
    print(y_train.value_counts())

    print("\nTesting target distribution:")
    print(y_test.value_counts())

    # Dataset diagnostic: laundering cases by date
    daily_summary = transactions.groupby(
        transactions["timestamp"].dt.date
    ).agg(
        total_transactions=("is_laundering", "size"),
        laundering_cases=("is_laundering", "sum"),
    )

    daily_summary["laundering_rate_percent"] = (
        daily_summary["laundering_cases"]
        / daily_summary["total_transactions"]
        * 100
    ).round(2)

    print("\nMissing values in training features:")
    print(X_train.isnull().sum())

    print("\nMissing values in testing features:")
    print(X_test.isnull().sum())

    print("\nDaily laundering rate:")
    print(daily_summary)
