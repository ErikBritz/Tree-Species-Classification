import unittest

import numpy as np

from src.metrics import classification_report_text, compute_metrics


class MetricsTests(unittest.TestCase):
    def test_metrics_keep_classes_missing_from_labels(self) -> None:
        class_names = ("alder", "birch")
        y_true = np.array([0, 0, 0])
        y_pred = np.array([0, 0, 0])

        metrics = compute_metrics(y_true, y_pred, class_names)
        self.assertEqual(metrics.macro_f1, 0.5)
        self.assertEqual(set(metrics.per_class_f1), set(class_names))
        self.assertEqual(metrics.per_class_f1["birch"], 0.0)
        self.assertEqual(metrics.confusion.shape, (2, 2))

    def test_report_includes_classes_with_no_support(self) -> None:
        report = classification_report_text(
            np.array([0, 0]),
            np.array([0, 0]),
            ("alder", "birch"),
        )
        self.assertIn("birch", report)
        self.assertIn("alder", report)


if __name__ == "__main__":
    unittest.main()
