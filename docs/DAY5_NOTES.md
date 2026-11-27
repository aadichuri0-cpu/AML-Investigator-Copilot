# Day 5 — AML Model Evaluation and Error Analysis

## 1. Objective

The objective of Day 5 was to understand the trained AML classification model,
investigate its prediction errors, evaluate feature choices, and select a
decision threshold using validation data.

## 2. Threshold Experiment

The balanced Logistic Regression model was evaluated on the validation set
using different probability thresholds.

Validation set:
- Legitimate transactions: 2,960
- Laundering transactions: 240
- Total: 3,200

| Threshold | False Positives | False Negatives | Precision | Recall | F1-score | Accuracy |
|-----------|-----------------|-----------------|-----------|--------|----------|----------|
| 0.3 | 238 | 2 | 50.00% | 99.17% | 66.48% | 92.50% |
| 0.4 | 221 | 2 | 51.85% | 99.17% | 68.10% | 93.03% |
| 0.5 | 213 | 7 | 52.24% | 97.08% | 67.93% | 93.13% |
| 0.6 | 209 | 13 | 52.06% | 94.58% | 67.16% | 93.06% |
| 0.7 | 194 | 22 | 52.91% | 90.83% | 66.87% | 93.25% |

### Threshold decision

Threshold 0.4 was selected because it achieved 99.17% validation recall
while producing fewer false positives than threshold 0.3.

The threshold represents the probability cutoff used to classify a
transaction as laundering. It is a decision setting, not a retrained model.

## 3. False-Positive Investigation

A false positive is a legitimate transaction incorrectly predicted as
laundering.

The validation confusion matrix at threshold 0.4 was:

```text
[[2739, 221],
 [   2, 238]]
```

The investigation examined false positives by payment format and
cross-border status.

Findings:
- ACH transactions accounted for 140 of the 221 false positives.
- Cross-border transactions accounted for 178 false positives.
- ACH and cross-border transactions together accounted for 113 false positives.

These are associations observed in this dataset. They do not establish
that ACH or cross-border activity causes money laundering.

## 4. Feature Experiment: payment_format

Two balanced Logistic Regression models were compared on the same validation
split and at the same threshold of 0.4.

### Model A — With payment_format

```text
[[2739, 221],
 [   2, 238]]
```

- Precision: 51.85%
- Recall: 99.17%
- F1-score: 68.10%
- Accuracy: 93.03%

### Model B — Without payment_format

```text
[[2412, 548],
 [  55, 185]]
```

- Precision: 25.24%
- Recall: 77.08%
- F1-score: 38.03%
- Accuracy: 81.16%

### Decision

The experiment showed that removing payment_format substantially worsened
the validation results.

Therefore, payment_format was retained in the current model.

## 5. Final Test Evaluation

After selecting the threshold using validation data, the balanced model
was retrained on the full training split.

The final test set was evaluated once at threshold 0.4.

Test confusion matrix:

```text
[[3394, 306],
 [   7, 293]]
```

Final test metrics:

- Precision: 48.91%
- Recall: 97.67%
- F1-score: 65.18%
- Accuracy: 92.17%

The model detects most laundering examples in this test set, but it also
produces a considerable number of false-positive alerts.

The test set must not be used for further threshold selection or feature
experiments.

## 6. Saved Model

The model artifact is stored at:

```text
backend/app/ml/artifacts/aml_model.joblib
```

The artifact was verified to load successfully.

Saved threshold: 0.4

The pipeline contains:
- Preprocessing
- Logistic Regression classifier

## 7. Day 5 Conclusion

Day 5 focused on model evaluation and understanding model behaviour,
rather than simply improving accuracy.

Completed:
- Threshold comparison
- False-positive investigation
- Feature-removal experiment
- Validation-based threshold selection
- Final test evaluation
- Saved-model verification

Important limitation:
These results describe performance on the available dataset. They should
not be interpreted as proof of real-world AML detection performance.
