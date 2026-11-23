
from pathlib import Path

import pandas as pd
from datasets import load_dataset


# Project root directory
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Where our sample will be saved
OUTPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "raw"
    / "aml_transactions_sample.csv"
)

# Our sampling targets
NORMAL_TARGET = 18_500
LAUNDERING_TARGET = 1_500


def download_aml_sample():

    print("Connecting to Hugging Face dataset...")

    dataset = load_dataset(
        "aaronzeller/small-aml-data",
        split="train",
        streaming=True
    )

    # Shuffle records using a buffer
    dataset = dataset.shuffle(
        seed=42,
        buffer_size=500_000
    )

    normal_records = []
    laundering_records = []

    processed = 0

    print("\nCollecting transactions...")

    for record in dataset:

        processed += 1

        label = record["is_laundering"]

        # Collect legitimate transactions
        if label is False and len(normal_records) < NORMAL_TARGET:
            normal_records.append(record)

        # Collect laundering transactions
        elif label is True and len(laundering_records) < LAUNDERING_TARGET:
            laundering_records.append(record)

        # Show progress every 500,000 records
        if processed % 500_000 == 0:
            print(
                f"Processed: {processed:,} | "
                f"Normal: {len(normal_records):,} | "
                f"Laundering: {len(laundering_records):,}"
            )

        # Stop when both targets are reached
        if (
            len(normal_records) == NORMAL_TARGET
            and len(laundering_records) == LAUNDERING_TARGET
        ):
            break

    print("\nSampling completed.")

    print(f"Normal transactions: {len(normal_records):,}")
    print(f"Laundering transactions: {len(laundering_records):,}")

    # Combine both groups
    all_records = normal_records + laundering_records

    # Convert records into a DataFrame
    df = pd.DataFrame(all_records)

    # Shuffle the final sample
    df = df.sample(
        frac=1,
        random_state=42
    ).reset_index(drop=True)

    # Save the sample
    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    df.to_csv(OUTPUT_PATH, index=False)

    print("\nDataset saved successfully!")
    print(f"Total rows: {len(df):,}")
    print(f"Saved to: {OUTPUT_PATH}")

    print("\nFinal label distribution:")
    print(df["is_laundering"].value_counts())


if __name__ == "__main__":
    download_aml_sample()
