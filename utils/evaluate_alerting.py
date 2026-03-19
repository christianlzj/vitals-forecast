import pandas as pd
import matplotlib.pyplot as plt

TARGET_VITALS = ['HR', 'RESP', 'SpO2']
ALERT_THRESHOLDS = {
    # source: https://www.massgeneralbrigham.org/en/patient-care/services-and-specialties/heart/conditions/bradycardia
    # source: https://www.ridgmountpractice.nhs.uk/pulse-oximeter#:~:text=Normal%20Readings%20*%20Oxygen%20Saturation%20(Sp02)%20%2D,%2D%2040%2D100.%20*%20Temp%20(centigrade)%20%2D%2036.5%2D37.5.
    'HR': {
        'warning':   {'low': 50, 'high': 110},
        'emergency': {'low': 40, 'high': 130}
    },
    # source: https://www.lung.org/blog/respiratory-rate-vital-signs#:~:text=The%20normal%20respiratory%20rate%20for,minute%20is%20cause%20for%20concern.
    'RESP': {
        'warning':   {'low': 12, 'high': 20},
        'emergency': {'low': 11, 'high': 26}
    },
    # source: https://www.ridgmountpractice.nhs.uk/pulse-oximeter#:~:text=Normal%20Readings%20*%20Oxygen%20Saturation%20(Sp02)%20%2D,%2D%2040%2D100.%20*%20Temp%20(centigrade)%20%2D%2036.5%2D37.5.

    'SpO2': {
        'warning':   {'low': 94, 'high': None},
        'emergency': {'low': 92, 'high': None}
    }
}

def classify_alert(values, vital):
    thresholds = ALERT_THRESHOLDS[vital]
    
    def is_emergency(v):
        lo, hi = thresholds['emergency']['low'], thresholds['emergency']['high']
        return (lo is not None and v <= lo) or (hi is not None and v >= hi)
    
    def is_warning(v):
        lo, hi = thresholds['warning']['low'], thresholds['warning']['high']
        return (lo is not None and v <= lo) or (hi is not None and v >= hi)

    if any(is_emergency(v) for v in values):
        return 'emergency'
    elif any(is_warning(v) for v in values):
        return 'warning'
    else:
        return 'stable'

def evaluate_alert_hitrate(predicted, actual, vital):
    pred_label   = classify_alert(predicted, vital)
    actual_label = classify_alert(actual, vital)
    
    return {
        "Vital": vital,
        "Predicted": pred_label,
        "Actual": actual_label,
        "Exact Hit": pred_label == actual_label,
        "Binary Hit": (pred_label == 'stable') == (actual_label == 'stable')  # at least got stable vs non-stable right
    }

def plot_confusion_matrix(alert_df, vital, labels=['stable', 'warning', 'emergency']):
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    sub = alert_df[alert_df["Vital"] == vital]
    
    cm = confusion_matrix(sub["Actual"], sub["Predicted"], labels=labels)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=labels)
    
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, colorbar=False, cmap='Blues')
    ax.set_title(f"Alert Confusion Matrix — {vital}")
    plt.tight_layout()
    plt.show()

def summarize_alert_hitrate(alert_results):
    from sklearn.metrics import classification_report
    alert_df = pd.DataFrame(alert_results)
    labels = ['stable', 'warning', 'emergency']
    
    print("=== Alert Hit Rate by Vital ===")
    summary = alert_df.groupby("Vital")["Exact Hit"].agg(['sum', 'count', 'mean'])
    summary.columns = ['Hits', 'Total', 'HitRate']
    summary['HitRate'] = summary['HitRate'].map('{:.1%}'.format)
    print(summary)

    print("\n=== Per-Vital Classification Report ===")
    for vital in TARGET_VITALS:
        sub = alert_df[alert_df["Vital"] == vital]
        print(f"\n--- {vital} ---")
        print(classification_report(
            sub["Actual"], sub["Predicted"],
            labels=labels,
            zero_division=0
        ))

    print("\n=== Confusion Matrices ===")
    for vital in TARGET_VITALS:
        plot_confusion_matrix(alert_df, vital)
    
    return alert_df