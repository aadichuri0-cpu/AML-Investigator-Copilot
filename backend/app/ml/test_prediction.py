
from backend.app.ml.model_loader import load_model
from backend.app.ml.prepare_data import prepare_data
import numpy as np
from backend.app.ml.prepare_data import load_processed_data



def test_prediction():

    # Load the saved model and threshold
    model, threshold = load_model()

    # Load our prepared dataset
    X_train, X_test, y_train, y_test = prepare_data()

    # Select one transaction from the training data
    transaction = X_train.iloc[[0]]

    # Get the actual label for comparison
    actual_label = y_train.iloc[0]

    print("\nTransaction Features:")
    print(transaction.T)

    # Predict the probability of laundering
    probability = model.predict_proba(transaction)[0][1]

        # Inspect which features influenced the prediction
    preprocessor = model.named_steps["preprocessing"]
    classifier = model.named_steps["classifier"]

    transformed_transaction = preprocessor.transform(transaction)

    # Convert sparse matrix to normal array if needed
    if hasattr(transformed_transaction, "toarray"):
        transformed_transaction = transformed_transaction.toarray()

    feature_names = preprocessor.get_feature_names_out()

    # Each feature's contribution to the model's raw score
    contributions = (
        transformed_transaction[0] * classifier.coef_[0]
    )

    # Find the 10 strongest positive contributions
    top_indices = np.argsort(contributions)[-10:][::-1]

    print("\nTop Positive Feature Contributions:")
    for index in top_indices:
        if contributions[index] > 0:
            print(
                feature_names[index],
                round(contributions[index], 4)
            )

        # Check payment format patterns in training data only
    X_train, X_test, y_train, y_test = prepare_data()

    training_data = X_train.copy()
    training_data["is_laundering"] = y_train

    print("\nTraining laundering rate by payment format:")
    print(
        training_data.groupby("payment_format")["is_laundering"]
        .mean()
        .sort_values(ascending=False)
    )

    # Check how many ACH transactions are legitimate vs laundering
    print("\nACH Transaction Label Breakdown:")

    ach_transactions = training_data[
        training_data["payment_format"] == "ACH"
    ]

    print(ach_transactions["is_laundering"].value_counts())

    print("\nTraining Laundering Rate by Cross-Border Status:")

    print(
        training_data.groupby("is_cross_border")["is_laundering"]
        .agg(["count", "mean"])
    )

    # Apply our selected threshold
    prediction = probability >= threshold

    # Display the results
    print("\nTransaction Prediction")
    print("--------------------")
    print("Laundering probability:", round(probability, 4))
    print("Decision threshold:", threshold)
    print("Predicted laundering:", prediction)
    print("Actual laundering:", actual_label)


if __name__ == "__main__":
    test_prediction()
