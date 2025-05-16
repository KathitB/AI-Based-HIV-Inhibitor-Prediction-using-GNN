import deepchem as dc
import tensorflow as tf
from sklearn.metrics import classification_report
from preprocess import feature_data
from model import build_gnn_model


# Load and featurize the dataset
train_data, test_data = feature_data("C:\\Internship\\DRUG_DISCOVERY_HIV_INHIBITOR\\dataset\\HIV.csv")

# Build the GNN model
model = build_gnn_model()

# Train the model
model.fit(train_data, nb_epoch=5)

# Evaluate on the test data
metric = dc.metrics.Metric(dc.metrics.roc_auc_score, mode="classification")
score = model.evaluate(test_data, [metric])
print("Test AUC Score:", score)

#y_true = test_data.y
#y_pred = model.predict(test_data) > 0.5
#print("\nClassification Report:\n", classification_report(y_true, y_pred))

# Save model
model.save_checkpoint(model_dir="app/outputs/model")





