from preprocess import feature_data
from train import train_model
from evaluate import evaluate_model

if __name__ == "__main__":
    print("Starting Drug Discovery Pipeline...")

    train_dataset, test_dataset = feature_data("C:\Internship\DRUG_DISCOVERY_HIV_INHIBITOR\dataset\HIV.csv")
    model, history = train_model(train_dataset, test_dataset)
    evaluate_model(model, test_dataset)

    print("Pipeline completed.")