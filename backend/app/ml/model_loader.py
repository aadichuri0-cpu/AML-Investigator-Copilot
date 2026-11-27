
from pathlib import Path
import joblib


# Find the saved model file
MODEL_PATH = Path(__file__).resolve().parent / "artifacts" / "aml_model.joblib"


def load_model():

    # Load the saved artifact
    artifact = joblib.load(MODEL_PATH)

    # Extract the trained pipeline
    model = artifact["model"]

    # Extract the selected threshold
    threshold = artifact["threshold"]

    return model, threshold


if __name__ == "__main__":

    model, threshold = load_model()

    print("Model loaded successfully!")
    print("Threshold:", threshold)
    print("Pipeline steps:", model.named_steps.keys())

