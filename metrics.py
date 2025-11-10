import numpy as np
from sklearn.metrics import accuracy_score


def compute_metrics(labels, predictions):
    live_mask = (labels == 0)
    spoof_mask = (labels == 1)
    
    live_count = np.sum(live_mask)
    spoof_count = np.sum(spoof_mask)

    spoof_predictions = predictions[spoof_mask]
    apcer = np.sum(spoof_predictions == 0) / spoof_count if spoof_count > 0 else 0
    
    live_predictions = predictions[live_mask]
    bpcer = np.sum(live_predictions == 1) / live_count if live_count > 0 else 0

    ace = (apcer + bpcer) / 2

    accuracy = accuracy_score(labels, predictions)

    return apcer, bpcer, ace, accuracy


def find_optimal_threshold(labels, probabilities, based_on):
    unique_probs = np.unique(probabilities)
    sorted_probs = np.sort(unique_probs)
    midpoints = (sorted_probs[:-1] + sorted_probs[1:]) / 2
    thresholds = np.concatenate([
        [unique_probs[0] - 1e-7],
        midpoints,
        [unique_probs[-1] + 1e-7]
    ])
    
    apcer_values = []
    bpcer_values = []
    ace_values = []
    accuracy_values = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        apcer, bpcer, ace, accuracy = compute_metrics(labels, predictions)
        apcer_values.append(apcer)
        bpcer_values.append(bpcer)
        ace_values.append(ace)
        accuracy_values.append(accuracy)

    apcer_values = np.array(apcer_values)
    bpcer_values = np.array(bpcer_values)
    ace_values = np.array(ace_values)
    accuracy_values = np.array(accuracy_values)

    if based_on == "apcer":
        optimal_idx = np.argmin(apcer_values)
    elif based_on == "bpcer":
        optimal_idx = np.argmin(bpcer_values)
    elif based_on == "ace":
        optimal_idx = np.argmin(ace_values)
    else:
        optimal_idx = np.argmin(accuracy_values)
    
    return thresholds[optimal_idx], apcer_values[optimal_idx], bpcer_values[optimal_idx], ace_values[optimal_idx], accuracy_values[optimal_idx]