from model.randomforest import RandomForest
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import pandas as pd
import numpy as np

def evaluate_chained_outputs(y_true, y_pred, level):
    # Extract prediction up to the requested level (1: y2, 2: y2_y3, 3: y2_y3_y4)
    def get_level(val, l):
        parts = str(val).split('_')
        return '_'.join(parts[:int(l)])
        
    y_true_level = [get_level(y, level) for y in y_true]
    y_pred_level = [get_level(y, level) for y in y_pred]
    
    acc = accuracy_score(y_true_level, y_pred_level)
    print(f"Accuracy at Level {level}: {acc:.2f}")
    
    # print classification report (Precision, Recall, F1)
    report = classification_report(y_true_level, y_pred_level, zero_division=0)
    print("Classification Report:")
    print(report)
    
    # print confusion matrix with classification names
    unique_labels = sorted(list(set(y_true_level).union(set(y_pred_level))))
    cm = confusion_matrix(y_true_level, y_pred_level, labels=unique_labels)
    cm_df = pd.DataFrame(cm, index=[f"True: {l}" for l in unique_labels], columns=[f"Pred: {l}" for l in unique_labels])
    print("Confusion Matrix:")
    print(cm_df.to_string())
    print("\n" + "="*50 + "\n")

def model_predict(data, df, name):
    # Here we need to call the methods related to the model e.g., random forest 
    rf_model = RandomForest(name, data.get_embeddings(), data.get_type())
    print(f"Training RandomForest model '{name}'...")
    rf_model.train(data)
    
    print("Predicting with RandomForest...")
    rf_model.predict(data.get_X_test())
    
    print("\n--- Chained Multi-Output Evaluation ---")
    evaluate_chained_outputs(data.get_type_y_test(), rf_model.predictions, level=1)
    evaluate_chained_outputs(data.get_type_y_test(), rf_model.predictions, level=2)
    evaluate_chained_outputs(data.get_type_y_test(), rf_model.predictions, level=3)

def model_evaluate(model, data):
    model.print_results(data)