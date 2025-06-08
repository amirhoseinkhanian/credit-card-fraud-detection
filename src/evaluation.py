import os
import time
import psutil
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, precision_recall_curve, average_precision_score, matthews_corrcoef

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process(os.getpid())
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def evaluate_model(model, X_test, y_test, model_name, output_file, balanced=True):
    """Evaluate a model, measure resources, and save metrics."""
    # Measure inference time and memory
    start_time = time.time()
    start_memory = get_memory_usage()
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
    inference_time = time.time() - start_time
    inference_memory = get_memory_usage() - start_memory

    # Calculate metrics
    report = classification_report(y_test, y_pred, output_dict=True)
    roc_auc = roc_auc_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    ap_score = average_precision_score(y_test, y_pred_proba) if y_pred_proba is not None else None
    mcc = matthews_corrcoef(y_test, y_pred)

    # Save metrics to file
    with open(output_file, 'a') as f:
        f.write(f"\n=== {model_name} {'(Balanced)' if balanced else '(Imbalanced)'} ===\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        if roc_auc is not None:
            f.write("\nROC-AUC Score: ")
            f.write(str(roc_auc))
        if ap_score is not None:
            f.write("\nAverage Precision Score: ")
            f.write(str(ap_score))
        f.write("\nMatthews Correlation Coefficient: ")
        f.write(str(mcc))
        f.write("\nInference Time (seconds): ")
        f.write(str(inference_time))
        f.write("\nInference Memory Usage (MB): ")
        f.write(str(inference_memory))
        f.write("\n" + "="*50 + "\n")

    print(f"\n{model_name} {'(Balanced)' if balanced else '(Imbalanced)'} Evaluation:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    if roc_auc is not None:
        print("ROC-AUC Score:", roc_auc)
    if ap_score is not None:
        print("Average Precision Score:", ap_score)
    print("Matthews Correlation Coefficient:", mcc)
    print("Inference Time (seconds):", inference_time)
    print("Inference Memory Usage (MB):", inference_memory)

    return {
        'y_pred_proba': y_pred_proba,
        'inference_time': inference_time,
        'inference_memory': inference_memory,
        'precision': report['1']['precision'],
        'recall': report['1']['recall'],
        'f1_score': report['1']['f1-score'],
        'roc_auc': roc_auc,
        'ap_score': ap_score,
        'mcc': mcc
    }

def plot_roc_curve(models, X_test, y_test, output_file, balanced=True):
    """Plot ROC curves for all models."""
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if y_pred_proba is not None:
            fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
            roc_auc = roc_auc_score(y_test, y_pred_proba)
            plt.plot(fpr, tpr, label=f"{model_name} (AUC = {roc_auc:.2f})")
    
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f"ROC Curve {'(Balanced)' if balanced else '(Imbalanced)'}")
    plt.legend(loc="lower right")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close('all')

def plot_precision_recall_curve(models, X_test, y_test, output_file, balanced=True):
    """Plot Precision-Recall curves for all models."""
    plt.figure(figsize=(10, 8))
    for model_name, model in models.items():
        y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None
        if y_pred_proba is not None:
            precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
            plt.plot(recall, precision, label=f"{model_name}")
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(f"Precision-Recall Curve {'(Balanced)' if balanced else '(Imbalanced)'}")
    plt.legend(loc="lower left")
    plt.grid(True)
    plt.savefig(output_file)
    plt.close('all')

def plot_timing_comparison(timing_data, output_file):
    """Plot bar chart comparing training and inference times."""
    model_names = list(timing_data.keys())
    training_times = [timing_data[model]['training_time'] for model in model_names]
    inference_times = [timing_data[model]['inference_time'] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, training_times, width, label='Training Time', color='skyblue')
    plt.bar(x + width/2, inference_times, width, label='Inference Time', color='lightcoral')
    plt.xlabel('Models')
    plt.ylabel('Time (seconds)')
    plt.title('Training and Inference Time Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close('all')

def plot_memory_comparison(memory_data, output_file):
    """Plot bar chart comparing memory usage."""
    model_names = list(memory_data.keys())
    training_memory = [memory_data[model]['training_memory'] for model in model_names]
    inference_memory = [memory_data[model]['inference_memory'] for model in model_names]

    x = np.arange(len(model_names))
    width = 0.35

    plt.figure(figsize=(12, 6))
    plt.bar(x - width/2, training_memory, width, label='Training Memory', color='lightgreen')
    plt.bar(x + width/2, inference_memory, width, label='Inference Memory', color='salmon')
    plt.xlabel('Models')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Training and Inference Memory Usage Comparison')
    plt.xticks(x, model_names, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close('all')

def plot_metrics_comparison(metrics_data, output_file):
    """Plot bar chart comparing performance metrics."""
    df = pd.DataFrame(metrics_data).T
    metrics = ['precision', 'recall', 'f1_score', 'roc_auc', 'ap_score', 'mcc']
    model_names = df.index

    x = np.arange(len(model_names))
    width = 0.1

    plt.figure(figsize=(14, 8))
    for i, metric in enumerate(metrics):
        plt.bar(x + i*width, df[metric], width, label=metric.capitalize())

    plt.xlabel('Models')
    plt.ylabel('Score')
    plt.title('Performance Metrics Comparison')
    plt.xticks(x + width*2.5, model_names, rotation=45)
    plt.legend()
    plt.grid(True, axis='y')
    plt.tight_layout()
    plt.savefig(output_file)
    plt.close('all')

def compare_models(models, X_test, y_test, output_file, roc_output, pr_output, balanced=True):
    """Compare multiple models, save metrics, and plot curves."""
    if os.path.exists(output_file):
        os.remove(output_file)  # Clear previous results

    results = {}
    for model_name, model in models.items():
        result = evaluate_model(model, X_test, y_test, model_name, output_file, balanced)
        results[model_name] = result

    # Plot ROC and Precision-Recall curves
    plot_roc_curve(models, X_test, y_test, roc_output, balanced)
    plot_precision_recall_curve(models, X_test, y_test, pr_output, balanced)

    return results