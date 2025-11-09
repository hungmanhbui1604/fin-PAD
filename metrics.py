import numpy as np
from sklearn.metrics import accuracy_score


def compute_apcer_bpcer(targets, predictions):
    live_mask = (targets == 0)
    spoof_mask = (targets == 1)
    
    live_count = np.sum(live_mask)
    spoof_count = np.sum(spoof_mask)

    spoof_predictions = predictions[spoof_mask]
    apcer = np.sum(spoof_predictions == 0) / spoof_count if spoof_count > 0 else 0
    
    live_predictions = predictions[live_mask]
    bpcer = np.sum(live_predictions == 1) / live_count if live_count > 0 else 0

    return apcer, bpcer


def find_optimal_threshold_for_ace(targets, probabilities):
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
    accuracy_values = []
    
    for threshold in thresholds:
        predictions = (probabilities >= threshold).astype(int)
        apcer, bpcer = compute_apcer_bpcer(targets, predictions)
        apcer_values.append(apcer)
        bpcer_values.append(bpcer)
        accuracy_values.append(accuracy_score(targets, predictions))
    
    apcer_values = np.array(apcer_values)
    bpcer_values = np.array(bpcer_values)
    
    ace = (apcer_values + bpcer_values) / 2
    acc = 1 - ace
    optimal_idx = np.argmin(ace)
    
    return thresholds[optimal_idx], apcer_values[optimal_idx], bpcer_values[optimal_idx], accuracy_values[optimal_idx], ace[optimal_idx], acc[optimal_idx]