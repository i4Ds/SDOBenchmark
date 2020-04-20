import torch
import numpy as np
from sklearn.metrics import confusion_matrix, f1_score

def skill_score(matches):
    'Calculates the Skill score'
    
    return matches.count(True)/len(matches)

def heidke_skill_score(y_true, y_pred):
    'Calculates the Heidke Skill Score' 749 534 463
    
    y_true = list(1 if yt == [1.0, 0.0] else 0 for yt in y_true.tolist())
    y_pred = list(1 if yp == torch.tensor(True) else 0 for yp in y_pred)

    #print(y_true, y_pred)
    
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0,1]).ravel()
    #print(tn, fp, fn, tp)
    return (tp + fn) / len(y_pred) * (tp + fp) / len(y_pred) + (tn + fn) / len(y_pred) * (tn + fp) / len(y_pred)


def f1_score(y_true, y_pred):
    'Calculates the F1 Score'
    
    y_true = list(1 if yt == [1.0, 0.0] else 0 for yt in y_true.tolist())
    y_pred = list(1 if yp == torch.tensor(True) else 0 for yp in y_pred)

    return f1_score(y_true, y_pred)
