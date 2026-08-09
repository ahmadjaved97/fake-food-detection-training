# Metrics

Validation metrics are computed after gathering predictions, labels, and probabilities across all Accelerate processes.

Logged metrics:

- train loss
- train accuracy
- validation loss
- accuracy
- macro F1
- weighted F1
- macro precision
- weighted precision
- macro recall
- weighted recall
- per-class precision
- per-class recall
- per-class F1
- confusion matrix
- learning rate
- epoch time
- throughput in images per second

Macro metrics weight each class equally and are important for imbalanced data. Weighted metrics account for class frequency. Accuracy can be misleading when `real_bad` has few examples.

The default best checkpoint monitors validation `macro_f1`, so the default best checkpoint is named `best_macro_f1.pt`.

Multiclass ROC-AUC and PR-AUC can be enabled with:

```yaml
metrics:
  enable_auc: true
```

AUC metrics require all relevant classes to be present in the validation labels.
