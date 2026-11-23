# Day 2 — AML Transaction Data: EDA Findings

## 1. Objective

The objective of Day 2 was to inspect, validate, and explore the AML transaction dataset before proceeding to machine learning development.

The analysis focused on data quality, account activity, cross-border transactions, and payment formats.

## 2. Dataset Overview

- Total records: 20,000
- Total columns: 17
- Missing values: 0
- Duplicate rows: 0
- Duplicate transaction IDs: 0
- Invalid timestamps: 0

The dataset contains transaction details, sender and receiver information, transaction amounts, currencies, payment formats, and existing AML-related labels.

## 3. Data Validation Findings

| Check | Result |
|---|---:|
| Zero-value transactions | 211 |
| Negative-value transactions | 0 |
| Invalid timestamps | 0 |
| Same sender and receiver account ID | 12,812 |
| Cross-border transactions | 6,141 |

Zero-value transactions were identified for further investigation.

The high number of transactions with matching sender and receiver account IDs requires additional investigation. These records should not automatically be treated as invalid without understanding the dataset's construction and account identifiers.

## 4. Account Activity Analysis

Sender accounts were grouped to calculate:

- Transaction count
- Number of unique receivers
- Number of transactions labelled as laundering
- Laundering-label rate

Accounts with at least 10 transactions were shortlisted to reduce the effect of very small transaction counts.

Some shortlisted accounts had a 100% laundering-label rate in this sample.

These findings are exploratory. A high label rate alone does not establish that an account is suspicious in a real-world investigation.

## 5. Cross-Border Transaction Analysis

| Transaction type | Transactions | Labelled laundering | Label rate |
|---|---:|---:|---:|
| Domestic | 13,859 | 206 | 1.49% |
| Cross-border | 6,141 | 1,294 | 21.07% |

Cross-border transactions had a substantially higher proportion of laundering labels in this sample.

This is an observed association, not proof that cross-border transactions cause money laundering. Cross-border activity alone should not be treated as suspicious.

## 6. Payment Format Analysis

| Payment format | Transactions | Labelled laundering | Label rate |
|---|---:|---:|---:|
| ACH | 1,975 | 1,096 | 55.49% |
| Cash | 597 | 59 | 9.88% |
| Cheque | 2,348 | 185 | 7.88% |
| Bitcoin | 772 | 50 | 6.48% |
| Credit Card | 1,861 | 110 | 5.91% |
| Reinvestment | 12,226 | 0 | 0.00% |
| Wire | 221 | 0 | 0.00% |

ACH had the highest laundering-label rate in the selected sample.

Reinvestment and Wire had no transactions labelled as laundering in this sample. This does not mean these payment formats cannot be involved in laundering.

## 7. Transaction Amount Observations

The dataset's overall `amount_paid` summary showed:

- Mean: approximately 24.2 million
- Median: approximately 2,644
- Maximum: approximately 272.7 billion

The mean is much larger than the median, suggesting that extremely large transaction values influence the average.

The dataset contains multiple currencies. These aggregate amount statistics should therefore be interpreted cautiously and should not be treated as a single currency's monetary distribution.

Further analysis should compare amounts within currency groups.

## 8. Important Dataset Limitations

- The sample contains 20,000 transactions and is enriched with laundering-positive examples.
- The observed label rates are not real-world laundering prevalence estimates.
- Existing columns such as `predicted_alert` and `model_score` may contain outputs from a previous model or process. Their meaning must be investigated before using them as features.
- Currency differences must be handled when analyzing transaction amounts.
- Exploratory patterns do not establish causation or prove criminal activity.

## 9. Conclusion

Day 2 established a basic understanding of the dataset's structure, data quality, account activity, cross-border patterns, payment formats, and transaction amounts.

The dataset is ready for further investigation, subject to the limitations and follow-up checks documented above.

The next phase will build on these findings and continue the AML Copilot development process.
