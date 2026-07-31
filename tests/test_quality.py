"""``pgs_recon.utils.quality.detect_outliers``.

``pgs-quality-check`` flags anomalous capture positions in a scan's exposure
series through this wrapper. Every case below is a way the wrapper used to fail
outright -- an unfitted estimator, ``predict`` on a non-novelty LOF, a series
shaped as one sample of N features, a series shorter than the neighbor count --
so this is a regression guard, not a test of LOF itself.

Skips when scikit-learn is absent (``quality`` imports it at module load).
"""
import importlib.util
import unittest

MISSING = [d for d in ('numpy', 'sklearn')
           if importlib.util.find_spec(d) is None]

# A tight exposure series with one blown-out position (index 4).
SERIES = [120., 122., 119., 121., 240., 118., 123., 120., 119., 121., 122., 120.]


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class TestDetectOutliers(unittest.TestCase):

    @staticmethod
    def as_samples(series):
        """The orientation callers must use: one sample per measurement."""
        import numpy as np
        return np.asarray(series, dtype=float).reshape(-1, 1)

    def detect(self, series, **kwargs):
        from pgs_recon.utils.quality import detect_outliers
        return detect_outliers(self.as_samples(series), **kwargs)

    def test_labels_are_per_measurement(self):
        y, _ = self.detect(SERIES, n_neighbors=6)
        self.assertEqual(len(y), len(SERIES))

    def test_flags_the_anomalous_position(self):
        import numpy as np
        y, _ = self.detect(SERIES, n_neighbors=6)
        self.assertIn(4, np.where(y == -1)[0].tolist())

    def test_scores_align_with_labels(self):
        """The caller indexes negative_outlier_factor_ by label position."""
        import numpy as np
        y, clf = self.detect(SERIES, n_neighbors=6)
        scores = clf.negative_outlier_factor_
        self.assertEqual(len(scores), len(y))
        flagged = np.where(y == -1)[0]
        # The blown-out position must score worse than any inlier.
        self.assertLess(scores[4], scores[y == 1].min())
        self.assertTrue(all(scores[i] < 0 for i in flagged))

    def test_reused_estimator_is_refitted(self):
        """pgs-quality-check threads one clf through every series; each must be
        fitted on the series being labelled, not the first one seen."""
        y1, clf = self.detect(SERIES, n_neighbors=6)
        other = [50.0] * 6 + [300.0] + [50.0] * 5
        y2, clf2 = self.detect(other, clf=clf)
        self.assertIs(clf2, clf)
        self.assertEqual(len(y2), len(other))
        self.assertIn(6, [i for i, v in enumerate(y2) if v == -1])

    def test_short_series_reports_all_inlier(self):
        """Too short for LOF: a QC plot should render, not crash."""
        for series in ([120.0, 240.0], [120.0], []):
            with self.subTest(n=len(series)):
                y, _ = self.detect(series)
                self.assertEqual(len(y), len(series))
                self.assertTrue(all(v == 1 for v in y))

    def test_series_shorter_than_neighbors_reports_all_inlier(self):
        y, _ = self.detect([120.0, 121.0, 119.0, 240.0], n_neighbors=6)
        self.assertTrue(all(v == 1 for v in y))

    def test_unfitted_estimator_is_not_left_for_the_caller_to_touch(self):
        """When nothing is fitted, no label comes back -1, so the caller never
        reaches negative_outlier_factor_ (which would not exist yet)."""
        y, clf = self.detect([120.0, 240.0])
        self.assertFalse(any(v == -1 for v in y))
        self.assertFalse(hasattr(clf, 'negative_outlier_factor_'))


if __name__ == '__main__':
    unittest.main()
