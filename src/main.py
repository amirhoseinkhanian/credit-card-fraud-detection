import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
import time
import psutil
from data_preprocessing import load_data, explore_data, preprocess_data_balanced, preprocess_data_imbalanced
from model_training import train_random_forest, train_xgboost, train_logistic_regression
from evaluation import compare_models, plot_timing_comparison, plot_memory_comparison, plot_metrics_comparison

# Set plotting style
plt.style.use('ggplot')

def get_memory_usage():
    """Get current process memory usage in MB."""
    process = psutil.Process()
    return process.memory_info().rss / (1024 * 1024)  # Convert bytes to MB

def main():
    """Main function to run the fraud detection pipeline."""
    # Define paths
    data_path = 'data/creditcard.csv'
    output_file = 'results/model_comparison.txt'
    params_file = 'results/best_params.json'
    roc_output = 'results/roc_curve.png'
    pr_output = 'results/precision_recall_curve.png'
    roc_imbalanced_output = 'results/roc_curve_imbalanced.png'
    pr_imbalanced_output = 'results/precision_recall_imbalanced.png'
    timing_output = 'results/timing_comparison.png'
    memory_output = 'results/memory_usage.png'
    metrics_output = 'results/metrics_comparison.png'

    # Load and explore data
    data = load_data(data_path)
    explore_data(data)

    # Preprocess data for balanced models (with SMOTE)
    X_train_balanced, X_test, y_train_balanced, y_test = preprocess_data_balanced(data)

    # Preprocess data for imbalanced models (without SMOTE)
    X_train_imbalanced, X_test_imbalanced, y_train_imbalanced, y_test_imbalanced = preprocess_data_imbalanced(data)

    # Ensure test sets are identical
    assert X_test.equals(X_test_imbalanced) and y_test.equals(y_test_imbalanced), "Test sets must be identical"

    # Initialize data structures
    models = {}
    models_imbalanced = {}
    timing_data = {}
    memory_data = {}
    metrics_data = {}

    # Train and evaluate models on balanced data
    for model_name, train_func in [
        ('Random Forest', train_random_forest),
        ('XGBoost', train_xgboost),
        ('Logistic Regression', train_logistic_regression)
    ]:
        # Measure training time and memory
        start_time = time.time()
        start_memory = get_memory_usage()
        model = train_func(X_train_balanced, y_train_balanced, params_file, 'balanced')
        training_time = time.time() - start_time
        training_memory = get_memory_usage() - start_memory

        models[model_name] = model
        timing_data[model_name] = {'training_time': training_time}
        memory_data[model_name] = {'training_memory': training_memory}

    # Compare models on balanced data
    results = compare_models(models, X_test, y_test, output_file, roc_output, pr_output, balanced=True)

    # Update timing, memory, and metrics for balanced models
    for model_name, result in results.items():
        timing_data[model_name]['inference_time'] = result['inference_time']
        memory_data[model_name]['inference_memory'] = result['inference_memory']
        metrics_data[model_name] = {
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'roc_auc': result['roc_auc'],
            'ap_score': result['ap_score'],
            'mcc': result['mcc']
        }

    # Train and evaluate models on imbalanced data
    for model_name, train_func in [
        ('Random Forest', train_random_forest),
        ('XGBoost', train_xgboost),
        ('Logistic Regression', train_logistic_regression)
    ]:
        # Measure training time and memory
        start_time = time.time()
        start_memory = get_memory_usage()
        model = train_func(X_train_imbalanced, y_train_imbalanced, params_file, 'imbalanced')
        training_time = time.time() - start_time
        training_memory = get_memory_usage() - start_memory

        models_imbalanced[model_name] = model
        timing_data[f"{model_name} (Imbalanced)"] = {'training_time': training_time}
        memory_data[f"{model_name} (Imbalanced)"] = {'training_memory': training_memory}

    # Compare models on imbalanced data
    results_imbalanced = compare_models(models_imbalanced, X_test, y_test, output_file, roc_imbalanced_output, pr_imbalanced_output, balanced=False)

    # Update timing, memory, and metrics for imbalanced models
    for model_name, result in results_imbalanced.items():
        timing_data[f"{model_name} (Imbalanced)"]['inference_time'] = result['inference_time']
        memory_data[f"{model_name} (Imbalanced)"]['inference_memory'] = result['inference_memory']
        metrics_data[f"{model_name} (Imbalanced)"] = {
            'precision': result['precision'],
            'recall': result['recall'],
            'f1_score': result['f1_score'],
            'roc_auc': result['roc_auc'],
            'ap_score': result['ap_score'],
            'mcc': result['mcc']
        }

    # Plot comparisons
    plot_timing_comparison(timing_data, timing_output)
    plot_memory_comparison(memory_data, memory_output)
    plot_metrics_comparison(metrics_data, metrics_output)

    print(f"\nModel comparison results saved to {output_file}")
    print(f"Best parameters saved to {params_file}")
    print(f"ROC curve saved to {roc_output} and {roc_imbalanced_output}")
    print(f"Precision-Recall curve saved to {pr_output} and {pr_imbalanced_output}")
    print(f"Timing comparison saved to {timing_output}")
    print(f"Memory usage comparison saved to {memory_output}")
    print(f"Metrics comparison saved to {metrics_output}")

if __name__ == "__main__":
    main()