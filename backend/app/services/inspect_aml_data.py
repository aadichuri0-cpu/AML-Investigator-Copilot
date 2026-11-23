
from pathlib import Path

import pandas as pd




PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aml_transactions_sample.csv"
)


def validate_transactions(df):
    """
    Check the AML transaction dataset for basic data quality issues.
    """

    print("\n========== DATA VALIDATION ==========")

    # 1. Check whether transaction IDs are duplicated
    duplicate_ids = df["record_key"].duplicated().sum()
    print(f"\nDuplicate transaction IDs: {duplicate_ids}")

    # 2. Check transaction amounts
    zero_amounts = (df["amount_paid"] == 0).sum()
    negative_amounts = (df["amount_paid"] < 0).sum()

    print(f"Zero-value transactions: {zero_amounts}")
    print(f"Negative-value transactions: {negative_amounts}")

    # 3. Check whether timestamps can be converted to datetime
    parsed_timestamps = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
        utc=True
    )

    invalid_timestamps = parsed_timestamps.isna().sum()
    print(f"Invalid timestamps: {invalid_timestamps}")

    # 4. Check transactions where sender and receiver are the same
    same_account = (
        df["from_account"] == df["to_account"]
    ).sum()

    print(f"Same sender and receiver account: {same_account}")

    # 5. Identify cross-border transactions
    cross_border = (
        df["from_country"] != df["to_country"]
    ).sum()

    print(f"Cross-border transactions: {cross_border}")

    print("\nValidation completed.")



def inspect_dataset():

    df = pd.read_csv(DATA_PATH)

    # Display all columns instead of shortening the table
    pd.set_option("display.max_columns", None)
    pd.set_option("display.width", 200)

    print("\n========== DATASET OVERVIEW ==========")

    print("\nShape:")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types and non-null counts:")
    df.info()

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nDuplicate rows:")
    print(df.duplicated().sum())

    print("\nStatistical summary:")
    print(df.describe(include="all"))

    return df


def analyze_account_activity(df):
    """
    Analyse transaction activity for each sender account.
    """

    print("\n========== ACCOUNT ACTIVITY ANALYSIS ==========")

    account_activity = (
        df.groupby("from_account")
        .agg(
            transaction_count=("record_key", "count"),
            unique_receivers=("to_account", "nunique"),
            laundering_transactions=("is_laundering", "sum")
        )
        .reset_index()
    )

    # Calculate laundering-label percentage
    account_activity["laundering_rate"] = (
        account_activity["laundering_transactions"]
        / account_activity["transaction_count"]
    ) * 100

    # Keep accounts with at least 10 transactions
    active_accounts = account_activity[
        account_activity["transaction_count"] >= 10
    ]

    # Sort accounts by laundering-label rate
    active_accounts = active_accounts.sort_values(
        by="laundering_rate",
        ascending=False
    )

    print("\nAccounts with at least 10 transactions:")
    print(active_accounts.head(10))

def analyze_cross_border_activity(df):

    """
    Compare domestic and cross-border transactions.
    """

    # Create a column to identify cross-border transactions
    df = df.copy()

    df["cross_border"] = (
        df["from_country"] != df["to_country"]
    )

    # Group transactions into domestic and cross-border
    analysis = df.groupby("cross_border").agg(
        transaction_count=("record_key", "count"),
        laundering_transactions=("is_laundering", "sum")
    )

    # Calculate the percentage of laundering-labelled transactions
    analysis["laundering_rate"] = (
        analysis["laundering_transactions"]
        / analysis["transaction_count"]
    ) * 100

    print("\nCross-Border Transaction Analysis:")
    print(analysis)

def analyze_payment_formats(df):

    print("\n========== PAYMENT FORMAT ANALYSIS ==========")

    payment_analysis = df.groupby("payment_format").agg(
        transaction_count=("record_key", "count"),
        laundering_transactions=("is_laundering", "sum")
    )

    payment_analysis["laundering_rate"] = (
        payment_analysis["laundering_transactions"]
        / payment_analysis["transaction_count"]
    ) * 100

    payment_analysis = payment_analysis.sort_values(
        by="laundering_rate",
        ascending=False
    )

    print(payment_analysis)

if __name__ == "__main__":

    df = inspect_dataset()

    validate_transactions(df)

    analyze_account_activity(df)

    analyze_cross_border_activity(df)

    analyze_payment_formats(df)