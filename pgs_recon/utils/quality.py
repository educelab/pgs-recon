import numpy as np
from sklearn import neighbors


def measure_exposure(img):
    """Adapted from:
    Romaniak, Piotr, et al. "A no reference metric for the quality assessment
    of videos affected by exposure distortion." 2011 IEEE International
    Conference on Multimedia and Expo. IEEE, 2011.
    @param img:
    @return:
    """
    ys = int(np.ceil(img.shape[0] / 4))
    xs = int(np.ceil(img.shape[1] / 4))
    l_vals = []
    for y in range(0, img.shape[0], ys):
        for x in range(0, img.shape[1], xs):
            yy = min(img.shape[0], y + ys)
            xx = min(img.shape[1], x + xs)
            l_vals.append(np.mean(img[y:yy, x:xx]))
    l_vals = sorted(l_vals)
    ll = np.mean(l_vals[:3])
    lu = np.mean(l_vals[-3:])
    return (lu + ll) / 2


def detect_outliers(x, clf=None, **kwargs):
    """Label each row of ``x`` inlier (1) or outlier (-1) by Local Outlier Factor.

    ``x`` is ``(n_samples, n_features)``; a 1-D series of measurements is
    therefore ``series.reshape(-1, 1)`` -- one sample per measurement -- so that
    the labels come back per measurement.

    Always (re)fits on the data being labelled. LOF's labels and its
    ``negative_outlier_factor_`` scores both describe the set the estimator was
    fitted on, so scoring one series with a model fitted on another says nothing
    about it; and ``predict`` does not even exist unless the estimator was
    constructed with ``novelty=True``, which is why this uses ``fit_predict``.
    ``clf`` lets a caller reuse the estimator object across series (it is refitted
    each time); ``kwargs`` configure a new one and are ignored when ``clf`` is
    given.

    Returns ``(labels, clf)``, where ``clf.negative_outlier_factor_`` holds the
    per-sample scores aligned with ``labels`` -- but only once something has been
    fitted, so read it only where a label came back ``-1``. A series too short for
    LOF (under 3 samples, or no more samples than neighbors) is reported
    all-inlier rather than raising: a quality plot should say "nothing anomalous
    found" instead of aborting the run.
    """
    n = len(x)
    if clf is None:
        kwargs.setdefault('n_neighbors', 8)
        kwargs['n_neighbors'] = min(kwargs['n_neighbors'], max(n - 1, 1))
        clf = neighbors.LocalOutlierFactor(**kwargs)
    if n < 3 or clf.n_neighbors >= n:
        return np.ones(n, dtype=int), clf
    return clf.fit_predict(x), clf


def main():
    from pathlib import Path
    from tqdm import tqdm
    import imageio.v3 as iio
    import pickle
    import argparse
    import logging
    import sys
    import json

    logger = logging.getLogger('pgs-classify-outliers')

    parser = argparse.ArgumentParser('pgs-classify-outliers')
    parser.add_argument('-i', '--input-dir', required=True, nargs='+',
                        help='PGS dataset directories')
    parser.add_argument('-o', '--output-dir', required=True,
                        help='Output directory for results')
    args = parser.parse_args()

    all_exposures = []
    for d in tqdm(args.input_dir, 'datasets'):
        # Check if the directory is a scan dir
        scan_dir = Path(d)
        meta_path = scan_dir / 'metadata.json'
        if not meta_path.exists():
            logger.error(f'input is not a PGS dataset: {d}')
            sys.exit(-1)

        # Load the metadata
        with meta_path.open() as f:
            meta = json.loads(f.read())

        # Get the images list
        prefix = meta['scan']['file_prefix']
        ext = meta['scan']['format'].lower()
        images = list(scan_dir.glob(f'{prefix}*.{ext}'))
        images.sort()

        # Load and measure the images
        exposures = []
        for i in tqdm(images, f' - {d[:15]}...', leave=False):
            img = iio.imread(i)
            exposures.append(measure_exposure(img))
        all_exposures.append(exposures)

    print('fitting model')
    all_exposures = np.array(all_exposures)
    clf = neighbors.LocalOutlierFactor(n_neighbors=5, novelty=True)
    clf = clf.fit(all_exposures)

    print('saving model')
    out_dir = Path(args.output_dir)
    with (out_dir / 'ir_classifier_8bpc.pkl').open('wb') as of:
        pickle.dump(clf, of)


if __name__ == '__main__':
    main()
