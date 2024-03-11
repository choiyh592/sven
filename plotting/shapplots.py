import shap
import numpy as np

def shap_waterfall_plot(show_idx, shap_values, shap_expectation, feature_names, max_display=10, show=True):
    explanation = shap.Explanation(shap_expectation, shap_values, feature_names=feature_names)
    shap.waterfall_plot(explanation[show_idx], max_display=max_display, show=show)