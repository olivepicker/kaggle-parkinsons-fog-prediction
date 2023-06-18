import torch
import librosa
import numpy as np
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, recall_score, average_precision_score
import torch.nn.functional as F

def compute_binary_scores(label, probability):
    score = average_precision_score(label, probability)
    roc = roc_auc_score(label, probability)
    
    label = np.asarray(label).flatten()
    probability = np.asarray(probability).flatten()
    
    f1  = f1_score(label, probability>0.5)
    acc = accuracy_score(label, probability>0.5) * 100
    recall = recall_score(label, probability>0.5)
    ret = {
        'acc' : acc,
        'roc' : roc,
        'f1'  : f1,
        'recall' : recall,
        'score' : score
    }
    return ret

def compute_multi_scores(label, probability):
    
    label = np.asarray(label).flatten()
    probability = np.asarray(probability).flatten()
    acc = accuracy_score(label, probability) * 100
    f1  = f1_score(label, probability, average = 'macro')
    recall = recall_score(label, probability, average = 'macro')
    ret = {
        'acc' : acc,
        #'roc' : roc,
        'f1'  : f1,
        'recall' : recall
    }
    return ret
