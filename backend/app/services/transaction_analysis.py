
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]

DATA_PATH = PROJECT_ROOT / "data" / "raw" / "aml_transactions_sample.csv"


def load_transactions():
    """
    Load transaction data from the CSV file.
    """

    df = pd.read_csv(DATA_PATH)

    df["timestamp"] = pd.to_datetime(df["timestamp"])

    return df



if __name__ == "__main__":

    transactions = load_transactions()

    print("\nDataset shape:")
    print(transactions.shape)

    print("\nFirst 5 transactions:")
    print(transactions.head())

    print("\nDataset information:")
    transactions.info()

    print("\nStatistical summary:")
    print(transactions.describe())

    print("\nMissing values:")
    print(transactions.isnull().sum())

    print("\nDuplicate transaction rows:")
    print(transactions.duplicated().sum())  

    print("\nDuplicate transaction IDs:")
    print(transactions["transaction_id"].duplicated().sum())

    print("\nTransactions with invalid amounts:")
    print((transactions["amount"] <= 0).sum())


