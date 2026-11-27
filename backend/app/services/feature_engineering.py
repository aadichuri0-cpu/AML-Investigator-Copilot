
from backend.app.services.transaction_analysis import load_transactions

from pathlib import Path

import os
import psutil


def create_transaction_features(df):
    """
    Create useful features from transaction data.
    """

    df = df.copy()

    # Feature 1: Identify cross-border transactions
    df["is_cross_border"] = (
        df["from_country"] != df["to_country"]
    )

    # Feature 2: Extract the transaction hour
    df["transaction_hour"] = df["timestamp"].dt.hour


    # Feature 3: Extract the day of the week
    df["transaction_day"] = df["timestamp"].dt.day_of_week

    # Feature 4: Identify weekend transactions
    df["is_weekend"] = df["transaction_day"] >= 5


    # Feature 5: Count transactions made by each sender
    df["sender_transaction_count"] = (
        df.groupby("from_account")["record_key"]
        .transform("count")
    )

    # Feature 6: Count unique receivers for each sender
    df["sender_unique_receivers"] = (
        df.groupby("from_account")["to_account"]
        .transform("nunique")
    )

    # Feature 7: Calculate total amount transferred
    # by each sender, separately for each currency

    df["sender_total_amount"] = (
        df.groupby(
            ["from_account", "payment_currency"]
        )["amount_paid"]
        .transform("sum")
    )



    return df

def validate_features(df):
    """
    Validate the engineered transaction dataset.
    """

    print("\n--- Feature Validation ---")

    # Check total number of rows
    print("Total rows:", len(df))

    # Check for missing values
    print("\nMissing values:")
    print(df.isnull().sum())

    # Check for duplicate transaction IDs
    print("\nDuplicate record keys:")
    print(df["record_key"].duplicated().sum())

    # Check whether engineered features exist
    print("\nEngineered features:")
    print(
        df[
            [
                "is_cross_border",
                "transaction_hour",
                "transaction_day",
                "is_weekend",
                "sender_transaction_count",
                "sender_unique_receivers",
                "sender_total_amount"
            ]
        ].head()
    )

    # Measure DataFrame memory usage
    memory_bytes = df.memory_usage(deep=True).sum()

    memory_mb = memory_bytes / (1024 ** 2)

    print(f"DataFrame RAM usage: {memory_mb:.2f} MB")

    # Measure current Python process memory
    process = psutil.Process(os.getpid())

    process_memory_mb = (
        process.memory_info().rss / (1024 ** 2)
    )

    print(f"Python process RAM usage: {process_memory_mb:.2f} MB")




if __name__ == "__main__":

    # Load raw transactions
    transactions = load_transactions()

    # Create engineered features
    featured_transactions = create_transaction_features(
        transactions
    )

    # Validate the dataset
    validate_features(featured_transactions)

    # Create output directory
    output_path = Path("data/processed/aml_features.csv")
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save processed dataset
    featured_transactions.to_csv(
        output_path,
        index=False
    )

    print("\nProcessed dataset saved to:")
    print(output_path)



