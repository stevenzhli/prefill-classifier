import numpy as np
import pandas as pd
from plotnine import *
from sklearn.calibration import calibration_curve

from calibration.scaler import _prep_data


def plot_classwise_calibration_curve(y_true, y_probas, classes):
    """Plot the one-vs-rest calibration curve for each class.

    For each class, bins predicted probabilities and compares the mean predicted probability to the observed class frequency within bin. 
    Points on the diagonal indicate per-class calibration.
    """
    y_true, y_probas, classes = _prep_data(y_true, y_probas, classes)
    n_classes = len(classes)
    
    # Elegant one-hot encoding that handles both binary and multi-class flawlessly
    y_true_binary = np.eye(n_classes)[y_true]
    
    ls_df = []
    for i in range(n_classes):
        prob_true, prob_pred = calibration_curve(
            y_true=y_true_binary[:, i],
            y_prob=y_probas[:, i],
            n_bins=10,
            strategy='quantile'
        )
        _class_df = pd.DataFrame({
            'mean_pred_proba': prob_pred,
            'frac_positive': prob_true,
            'class_label': classes[i]
        })
        ls_df.append(_class_df)

    df = pd.concat(ls_df, axis=0, ignore_index=True)

    p = (
        ggplot(df, aes(x='mean_pred_proba', y='frac_positive', color='class_label'))
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="grey", size=0.5)
        + geom_line(size=1)
        + geom_point(size=2)
        + labs(
            x='mean predicted probability',
            y='observed true probability',
            title='Calibration curve (one-vs-rest)',
            color='class label'
        )
        + xlim(0, 1) + ylim(0, 1)
        + theme_minimal()
        + theme(aspect_ratio=1, figure_size=(10,10))
    )

    return p

def plot_predictive_calibration_curve(y_true, y_probas, classes, n_bins=10):
    """
    Plots a professional Predictive Calibration Curve (Reliability Diagram).
    """
    y_true, y_probas, classes = _prep_data(y_true, y_probas, classes)
    n_classes = len(classes)
    theoretical_min = 1.0 / n_classes

    confidences = np.max(y_probas, axis=1)
    predictions = np.argmax(y_probas, axis=1)
    is_correct = (predictions == y_true).astype(int)
    
    # Use Pandas Quantile Binning instead of Uniform Binning
    df_raw = pd.DataFrame({'confidence': confidences, 'correct': is_correct})
    df_raw['bin'] = pd.qcut(df_raw['confidence'], q=n_bins, duplicates='drop')
    
    df_data = []
    total_samples = len(confidences)
    
    # Group by the dynamically created bins
    for _, group in df_raw.groupby('bin', observed=False):
        count = len(group)
        if count > 0:
            df_data.append({
                'mean_confidence': group['confidence'].mean(), 
                'accuracy': group['correct'].mean(), 
                'count': count,
                'weight': count / total_samples
            })
            
    df = pd.DataFrame(df_data)
    
    p = (
        ggplot(df, aes(x="mean_confidence", y="accuracy"))
                
        # Perfect calibration line
        + geom_abline(intercept=0, slope=1, linetype="dashed", color="grey", size=0.8)
        
        # Plot lines and size points by the number of samples
        + geom_line(size=1, color="black")
        + geom_point(size=2, color="black", alpha=0.8)
        
        # Formatting
        + scale_size_continuous(range=[2, 8], name="Sample Count")
        + labs(
            x="Mean Confidence", 
            y="Accuracy",
            title="Calibration curve (predictive)",
        )
        + xlim(theoretical_min, 1) + ylim(theoretical_min, 1)
        + theme_minimal()
        + theme(aspect_ratio=1, figure_size=(10,10))
    )
    
    return p