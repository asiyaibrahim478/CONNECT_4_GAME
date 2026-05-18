import matplotlib
matplotlib.use('Agg') # Set non-interactive backend
import pandas as pd
import os
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from connect_db import load_all_data

def preprocess_and_train():
    # 1. Load All Data (Original + History)
    df = load_all_data()
    if df is None:
        print("Failed to load data.")
        return None

    # 2. Preprocessing
    print("\nPreprocessing data...")
    if df.isnull().values.any():
        df = df.dropna()
    
    X = df.drop('winner', axis=1)
    y = df['winner']

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    print(f"Train set size: {X_train.shape[0]}, Test set size: {X_test.shape[0]}")

    # 3. Define Models
    models = {
        "Random_Forest": RandomForestClassifier(n_estimators=100, random_state=42),
        "Gradient_Boosting": GradientBoostingClassifier(random_state=42),
        "Logistic_Regression": LogisticRegression(max_iter=1000, random_state=42)
    }

    # Use absolute paths
    base_dir = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(base_dir, 'models')
    plots_dir = os.path.join(base_dir, 'plots')

    for dir_name in [models_dir, plots_dir]:
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)

    # 4. Train, Evaluate, and Visualize
    accuracies = {}
    print("\nStarting model training and evaluation...")
    for name, model in models.items():
        print(f"--- Training {name} ---")
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc
        
        # Save Model
        model_path = os.path.join(models_dir, f"{name}.pkl")
        joblib.dump(model, model_path)
        
        # 5. Confusion Matrix Visualization (Thread-safe plotting)
        fig = plt.figure(figsize=(8, 6))
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=model.classes_, yticklabels=model.classes_)
        plt.title(f'Confusion Matrix: {name}\nAccuracy: {acc:.4f}')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        
        plot_path = os.path.join(plots_dir, f"{name}_cm.png")
        fig.savefig(plot_path)
        plt.close(fig)

    # 6. Accuracy Comparison Chart
    fig = plt.figure(figsize=(10, 6))
    names = list(accuracies.keys())
    values = list(accuracies.values())
    ax = sns.barplot(x=names, y=values, palette='viridis')
    plt.ylim(0, 1.0)
    plt.title('Model Accuracy Comparison (Updated)')
    plt.ylabel('Accuracy Score')
    for i, v in enumerate(values):
        ax.text(i, v + 0.02, f'{v:.4f}', ha='center', fontweight='bold')
    
    comp_plot_path = os.path.join(plots_dir, "accuracy_comparison.png")
    fig.savefig(comp_plot_path)
    plt.close(fig)

    print("\nAll models retrained and visualized successfully!")
    return accuracies

if __name__ == "__main__":
    preprocess_and_train()
