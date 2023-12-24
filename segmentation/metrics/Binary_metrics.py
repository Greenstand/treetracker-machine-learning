import numpy as np

class BinarySegMetrics():
    """
    Binary Segmentation
    """
    def __init__(self):
        # two classes (foreground and background)
        self.n_classes = 2
        self.confusion_matrix = np.zeros((2, 2))
        # self.threshold = 0.5  # Threshold for converting probabilities to binary predictions

    def _fast_hist(self, label_true, label_pred):
        # label_pred = label_pred >= self.threshold  # Binarize predictions
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            2 * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def get_results(self):
        """Returns accuracy score evaluation result for binary segmentation."""
        hist = self.confusion_matrix
        tn, fp, fn, tp = hist.ravel()
        
        # Metrics for foreground
        foreground_total = tp + fn
        foreground_acc = tp / foreground_total if foreground_total > 0 else 0
        
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0
        f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

        iou_foreground = tp / (tp + fn + fp) if (tp + fn + fp) > 0 else 0
        iou_background = tn / (tn + fp + fn) if (tn + fp + fn) > 0 else 0
        mean_iou = (iou_foreground + iou_background) / 2

        #overall_acc = np.diag(hist).sum() / hist.sum()

        return {
            "Foreground Acc": foreground_acc,
            "Precision": precision,
            "Recall": recall,
            "F1 Score": f1_score,
            "IoU Foreground": iou_foreground,
            "IoU Background": iou_background,
            "Mean IoU": mean_iou
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))

    @staticmethod
    def to_str(results):
        string = "\n"
        for k, v in results.items():
            string += "%s: %f\n" % (k, v)
        return string