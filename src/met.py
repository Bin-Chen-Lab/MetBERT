from sklearn.metrics import matthews_corrcoef, confusion_matrix, classification_report, roc_auc_score, f1_score
from sklearn.metrics import accuracy_score, roc_curve, auc, ConfusionMatrixDisplay, precision_recall_fscore_support
import numpy as np
import os
import torch
import matplotlib.pyplot as plt


def all_metrics(preds, labels, probs, model_name):
    """
    According to scikit-learns convention, confusion matrix means:
    TN | FP   
    FN | TP
    The first column contains, negative predictions (TN and FN)
    The second column contains, positive predictions (TP and FP)
    the first row contains negative labels (TN and FP)
    the second row contains positive labels (FN and TP)
    the diagonal contains the number of correctly predicted labels
    In our case we are using labels  [1,0]
    TP | FN
    FP | TN
    
    Also f1 score here is micro if average is binary
    https://sebastianraschka.com/faq/docs/computing-the-f1-score.html
    """

    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average='binary')
    acc = accuracy_score(labels, preds)
    print("\n")
    print('Testing accuracy', acc)
    print('\nTesting f1', f1)
    print('\nTesting precision', precision)
    print('\nTesting recall', recall)
    print("\n")
    cm = confusion_matrix(labels, preds, labels=[1, 0])
    print(cm)
    cmd = ConfusionMatrixDisplay(cm, display_labels=['Cancer','Non-Cancer'])
    cmd.plot()
    # saving confusion matrix
    plt.savefig('cm'+str(model_name)+'.jpg')
    print("\n")
    print("Testing Stats")
    tp, fn, fp, tn = cm.ravel()
    print(f"{'True Ve':^7} | {'False Ne':^9} | {'False Ve':^9} | {'True Ne':^7}")
    print(f"{tp :^7} | {fn :^9} | {fp :^9} | {tn :^7}")
    print("\n")
    print(classification_report(labels, preds))
    print("\n")
    v_auc = roc_auc_score(labels, preds)
    print(f'ROC/AUC: {v_auc}')
    print("\n")
    mcc = matthews_corrcoef(labels, preds)
    print('Total MCC: %.3f' % mcc)
    print("\n")
