import shap
import numpy as np

from sklearn.svm import SVR
import matplotlib

matplotlib.use('Agg')  # must be set before importing pyplot
import matplotlib.pyplot as plt


def generate_model_explainability(model, X_train, X_test, stock_name):
    # Enhanced SHAP summary plot for tree models or models supporting SHAP natively
    try:
        explainer = shap.Explainer(model, X_train)
        shap_values = explainer(X_test)

        # For tree-based or linear models that return shap_values as shap.Explanation object
        shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", color=shap_values.abs.mean(0),
                          max_display=10)
        fig_summary = plt.gcf()
        fig_summary.set_size_inches(8, 6)  # Adjust size to ensure proper side-by-side display
        summary_base64 = figs_to_base64(fig_summary)
        plt.clf()
    except TypeError:
        # Handle case where SHAP does not work (like with SVM models)
        summary_base64 = None

    # Feature importance plot for tree-based models and linear models (including SVM and Linear Regression)
    try:
        # For tree-based models like Random Forest, Decision Trees
        importances = model.feature_importances_ if hasattr(model, 'feature_importances_') else None
        if importances is None:
            # For models like Linear Regression or SVM, use coefficients instead
            if hasattr(model, 'coef_'):
                importances = np.abs(model.coef_)
            else:
                raise AttributeError("Model does not support feature importance directly")

        features = X_train.columns
        indices = np.argsort(importances)[::-1]

        # Create a horizontal bar chart
        fig, ax = plt.subplots(figsize=(8, 6))  # Adjust size for side-by-side layout
        ax.barh(range(len(features)), importances[indices], align="center", color='skyblue')
        ax.set_yticks(range(len(features)))
        ax.set_yticklabels([features[i] for i in indices])
        ax.set_xlabel('Feature Importance')
        ax.set_title(f"Feature Importance - {stock_name}")

        # Add value labels to the bars
        for i, v in enumerate(importances[indices]):
            ax.text(v + 0.02, i, f'{v:.3f}', va='center', fontweight='bold')

        # Tight layout and better spacing
        plt.tight_layout()

        feature_base64 = figs_to_base64(fig)
        plt.clf()

    except AttributeError:
        # Handle case for SVM with non-linear kernel (rbf) or any other model without feature importance
        if isinstance(model, SVR) and model.kernel == 'rbf':
            # Use SHAP's KernelExplainer for non-linear models like rbf SVR
            explainer = shap.KernelExplainer(model.predict, X_train)
            shap_values = explainer.shap_values(X_test)

            # Check if shap_values is a list (e.g., for multi-class classification)
            if isinstance(shap_values, list):
                shap_values = shap_values[0]  # For regression, we expect a single array

            # Create SHAP summary plot for non-linear SVM
            shap.summary_plot(shap_values, X_test, show=False, plot_type="dot", color=shap_values.mean(0),
                              max_display=10)
            fig_summary = plt.gcf()
            fig_summary.set_size_inches(8, 6)  # Adjust size to ensure proper side-by-side display
            summary_base64 = figs_to_base64(fig_summary)
            plt.clf()

            feature_base64 = None  # No feature importance plot for non-linear SVM
        else:
            feature_base64 = None
            summary_base64 = None

    return summary_base64, feature_base64


def figs_to_base64(fig):
    """Converts a Matplotlib figure to a base64 string."""
    from io import BytesIO
    import base64
    buffer = BytesIO()
    fig.savefig(buffer, format='png')
    buffer.seek(0)
    img_base64 = base64.b64encode(buffer.read()).decode('utf-8')
    return img_base64
