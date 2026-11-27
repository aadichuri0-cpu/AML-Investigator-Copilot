
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

categorical_features = [
    "from_bank",
    "from_country",
    "to_bank",
    "to_country",
    "receiving_currency",
    "payment_currency",
    "payment_format",
]

numerical_features = [
    "amount_received",
    "amount_paid",
    "transaction_day",
]

boolean_features = [
    "is_cross_border"
]


preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
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


def prepare_features(X_train, X_test):

    # Learn the preprocessing rules from training data
    preprocessor.fit(X_train)

    # Apply those learned rules to training data
    X_train_processed = preprocessor.transform(X_train)

    # Apply the SAME rules to testing data
    X_test_processed = preprocessor.transform(X_test)

    return X_train_processed, X_test_processed

