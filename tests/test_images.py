"""``pgs_recon.utils.images``' in-process CIE L*a*b* decode.

``pgs-convert`` is the one reader in the pipeline with no binary in the loop, so
nothing downstream will catch a Lab file it mistakes for RGB: imageio hands back
the stored samples either way, and the run finishes, quietly, with garbage
colors. The cases below are the distinctions that silence depends on -- Lab from
RGB, one photometric from another, a file's white point from the spec default --
plus the two encodings a decoder can only get right by consulting TIFF 6.0 s23
rather than by inspection: a*/b* are *signed*, and at 16 bits scaled by 256.

Skips when the imaging stack is absent, like ``test_quality``/``test_calibrate``.
"""
import importlib.util
import tempfile
import unittest
from pathlib import Path

MISSING = [d for d in ('numpy', 'skimage', 'tifffile', 'imageio', 'cv2')
           if importlib.util.find_spec(d) is None]

#: WhitePoint as TIFF stores it: two RATIONALs, tag 318, type 5.
D65_TAG = (318, 5, 2, (3127, 10000, 3290, 10000), False)
D50_TAG = (318, 5, 2, (34567, 100000, 35850, 100000), False)

#: A spread of saturated and neutral colors -- the extremes are where a wrong
#: sign or scale on a*/b* stops looking like a rounding error.
COLORS = [(0, 0, 0), (255, 255, 255), (128, 128, 128), (220, 30, 40),
          (30, 220, 40), (40, 30, 220), (240, 240, 30), (10, 90, 130)]


def rgb_row():
    """The test colors as one 8-bit image row."""
    import numpy as np
    return np.array([COLORS], dtype=np.uint8)


def lab_row():
    """Those colors in true L*a*b* units, against D65."""
    from skimage.color import rgb2lab
    return rgb2lab(rgb_row() / 255.)


def encode(lab, photometric, dtype, signed=False):
    """Store ``lab`` the way a TIFF writer would, per TIFF 6.0 s23.

    ``signed`` returns the same bits in the signed container a file that
    declares SampleFormat = 2 for its a*/b* channels reads back as.

    Module level rather than a fixture method: ``test_retexture`` builds the
    same Lab TIFF, and importing a function is not importing a ``TestCase``.
    """
    import numpy as np
    from pgs_recon.utils.images import CIELAB, ICCLAB_L_MAX_16
    full = 255 if dtype == np.uint8 else 65535
    # ICCLab's 16-bit L* is the one scale that is not the full range
    l_max = full if (dtype == np.uint8 or photometric == CIELAB) \
        else ICCLAB_L_MAX_16
    light = np.round(lab[..., 0] * l_max / 100.).astype(dtype)
    chroma = lab[..., 1:] * (1 if dtype == np.uint8 else 256)
    if photometric == CIELAB:  # signed, in the unsigned container
        signed_t = np.int8 if dtype == np.uint8 else np.int16
        chroma = np.round(chroma).astype(signed_t).astype(dtype)
    else:  # ICCLab: unsigned, biased by half the range
        bias = 128 if dtype == np.uint8 else 32768
        chroma = np.round(chroma + bias).clip(0, full).astype(dtype)
    samples = np.dstack([light, chroma])
    if signed:
        samples = samples.view(np.int8 if dtype == np.uint8 else np.int16)
    return samples


class TiffCase(unittest.TestCase):
    """A temp directory to write fixture TIFFs into."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.addCleanup(self._tmp.cleanup)

    def write(self, samples, photometric='rgb', name=None, extratags=()):
        import tifffile
        path = self.tmp / f'{name or photometric}.tif'
        tifffile.imwrite(path, samples, photometric=photometric,
                         extratags=list(extratags))
        return path


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class LabDecodeTest(TiffCase):

    def assert_close(self, got, tol, msg):
        """``got`` is float sRGB; compare in the 8-bit units users see."""
        import numpy as np
        err = np.abs(np.round(got * 255).astype(int) - rgb_row().astype(int))
        self.assertLessEqual(err.max(), tol, f'{msg} (max err {err.max()}/255)')

    # -- what a file says it is -------------------------------------------

    def test_rgb_tiff_is_not_lab(self):
        """The whole feature hangs on not answering "Lab" for an RGB file."""
        from pgs_recon.utils.images import lab_encoding
        self.assertIsNone(lab_encoding(self.write(rgb_row(), 'rgb')))

    def test_non_tiff_is_not_lab(self):
        """Only TIFF records a colorspace; everything else is assumed sRGB."""
        from pgs_recon.utils.images import lab_encoding
        jpg = self.tmp / 'x.jpg'
        jpg.write_bytes(b'not really a jpeg')
        self.assertIsNone(lab_encoding(jpg))

    def test_unreadable_file_is_not_lab(self):
        """A broken file is the reader's problem to report, not the probe's."""
        from pgs_recon.utils.images import lab_encoding
        broken = self.tmp / 'broken.tif'
        broken.write_bytes(b'II*\x00garbage')
        self.assertIsNone(lab_encoding(broken))

    def test_reads_photometric_and_white_point(self):
        from pgs_recon.utils.images import CIELAB, ICCLAB, lab_encoding
        import numpy as np
        lab = lab_row()
        for photometric, name in ((CIELAB, 'cielab'), (ICCLAB, 'icclab')):
            got = lab_encoding(self.write(encode(lab, photometric, np.uint8),
                                          name, extratags=[D65_TAG]))
            self.assertEqual(got.photometric, photometric)
            self.assertAlmostEqual(got.white_point[0], 0.3127, places=4)
            self.assertAlmostEqual(got.white_point[1], 0.3290, places=4)

    def test_white_point_defaults_to_d50_when_untagged(self):
        """TIFF 6.0's default. Guessing D65 instead shifts every color."""
        from pgs_recon.utils.images import CIELAB, D50, lab_encoding
        import numpy as np
        path = self.write(encode(lab_row(), CIELAB, np.uint16), 'cielab')
        self.assertEqual(lab_encoding(path).white_point, D50)

    # -- the decode itself -------------------------------------------------

    def test_round_trip_by_encoding(self):
        """Each storage form must decode back to the color it was made from."""
        from pgs_recon.utils.images import (CIELAB, ICCLAB, LabEncoding,
                                            lab_to_rgb)
        import numpy as np
        lab = lab_row()
        d65 = LabEncoding(CIELAB, (0.3127, 0.3290))
        # 16-bit carries a*/b* at 1/256 and comes back near-exact, which is the
        # tight bound here. 8-bit quantizes them to whole units, worth up to 8
        # sRGB levels on a dark saturated color -- measured, and reached by
        # quantizing L*a*b* alone, so it bounds the encoding, not the decode.
        for photometric, dtype, tol in ((CIELAB, np.uint16, 1),
                                        (CIELAB, np.uint8, 8),
                                        (ICCLAB, np.uint16, 1),
                                        (ICCLAB, np.uint8, 8)):
            with self.subTest(photometric=photometric, dtype=dtype.__name__):
                got = lab_to_rgb(encode(lab, photometric, dtype),
                                 d65._replace(photometric=photometric))
                self.assert_close(got, tol, 'round trip')

    def test_icclab_16bit_lightness_tops_out_below_full_scale(self):
        """ICC puts L* = 100 at 0xff00. Reading it as full scale is a uniform
        darkening that a round trip encoded the same wrong way cannot see."""
        from pgs_recon.utils.images import (ICCLAB, ICCLAB_L_MAX_16,
                                            LabEncoding, lab_to_rgb)
        import numpy as np
        white = np.array([[[ICCLAB_L_MAX_16, 32768, 32768]]], dtype=np.uint16)
        got = lab_to_rgb(white, LabEncoding(ICCLAB, (0.3127, 0.3290)))
        np.testing.assert_array_equal(np.round(got * 255).astype(int), 255)

    def test_signed_samples_decode_like_unsigned_ones(self):
        """A file may declare SampleFormat = 2 for its signed a*/b* channels,
        and the reader then hands back the whole array signed. Same bits, so it
        must decode the same; refusing it instead fails a whole dataset."""
        from pgs_recon.utils.images import CIELAB, LabEncoding, lab_to_rgb
        import numpy as np
        lab, enc = lab_row(), LabEncoding(CIELAB, (0.3127, 0.3290))
        for dtype in (np.uint8, np.uint16):
            with self.subTest(dtype=dtype.__name__):
                np.testing.assert_array_equal(
                    lab_to_rgb(encode(lab, CIELAB, dtype), enc),
                    lab_to_rgb(encode(lab, CIELAB, dtype, True), enc))

    def test_float_samples_are_already_lab_units(self):
        """No integer encoding to undo -- scaling them would be the bug."""
        from pgs_recon.utils.images import CIELAB, LabEncoding, lab_to_rgb
        got = lab_to_rgb(lab_row(), LabEncoding(CIELAB, (0.3127, 0.3290)))
        self.assert_close(got, 1, 'float round trip')

    def test_signed_chroma_is_not_read_as_unsigned(self):
        """The failure this all guards: a* / b* read as unsigned bytes.

        Decoding CIELab samples as if they were ICCLab (or vice versa) is a
        128-unit chroma error, so any color with real chroma must move a lot.
        """
        from pgs_recon.utils.images import (CIELAB, ICCLAB, LabEncoding,
                                            lab_to_rgb)
        import numpy as np
        samples = encode(lab_row(), CIELAB, np.uint8)
        white = (0.3127, 0.3290)
        right = lab_to_rgb(samples, LabEncoding(CIELAB, white))
        wrong = lab_to_rgb(samples, LabEncoding(ICCLAB, white))
        self.assertGreater(np.abs(right - wrong).max() * 255, 100)

    def test_white_point_changes_the_result(self):
        """Honoring the tag has to mean something, or it is not honored."""
        from pgs_recon.utils.images import CIELAB, D50, LabEncoding, lab_to_rgb
        import numpy as np
        samples = encode(lab_row(), CIELAB, np.uint16)
        d65 = lab_to_rgb(samples, LabEncoding(CIELAB, (0.3127, 0.3290)))
        d50 = lab_to_rgb(samples, LabEncoding(CIELAB, D50))
        self.assertGreater(np.abs(d65 - d50).max() * 255, 10)

    # -- refusals ----------------------------------------------------------

    def test_itulab_is_refused_not_guessed(self):
        from pgs_recon.utils.images import ITULAB, LabEncoding, lab_to_rgb
        import numpy as np
        with self.assertRaises(ValueError):
            lab_to_rgb(encode(lab_row(), ITULAB, np.uint8),
                       LabEncoding(ITULAB, (0.3127, 0.3290)))

    def test_non_three_channel_is_refused(self):
        from pgs_recon.utils.images import CIELAB, LabEncoding, lab_to_rgb
        import numpy as np
        with self.assertRaises(ValueError):
            lab_to_rgb(np.zeros((4, 4), np.uint8),
                       LabEncoding(CIELAB, (0.3127, 0.3290)))

    def test_unsupported_sample_type_is_refused(self):
        from pgs_recon.utils.images import CIELAB, LabEncoding, lab_to_rgb
        import numpy as np
        with self.assertRaises(ValueError):
            lab_to_rgb(np.zeros((4, 4, 3), np.int32),
                       LabEncoding(CIELAB, (0.3127, 0.3290)))


@unittest.skipIf(MISSING, f'requires {", ".join(MISSING)}')
class ReadSRGBTest(TiffCase):
    """The single reader all three apps go through.

    It replaced ImageMagick, so the two conversions ImageMagick got right are
    pinned here as exact identities rather than as tolerances: a 16-bit sample
    becomes ``round(v / 257)`` and an 8-bit one survives untouched. Those are
    what make dropping the dependency a no-op for every input except Lab.
    """

    def test_8bit_rgb_is_a_plain_rescale(self):
        from pgs_recon.utils.images import read_srgb
        import numpy as np
        src = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        got = read_srgb(self.write(src))
        self.assertEqual(got.dtype, np.uint8)
        self.assertFalse(got.decoded)
        np.testing.assert_allclose(got.pixels, src / 255.)

    def test_16bit_greyscale_keeps_its_single_channel(self):
        """Only some callers want three; forcing it here would surprise the rest."""
        from pgs_recon.utils.images import read_srgb
        import numpy as np
        src = np.arange(16, dtype=np.uint16).reshape(4, 4) * 4096
        got = read_srgb(self.write(src, photometric='minisblack'))
        self.assertEqual(got.pixels.shape, (4, 4))
        self.assertEqual(got.dtype, np.uint16)

    def test_16bit_rgb_png_keeps_its_depth_and_channel_order(self):
        """Pillow cannot hold 16-bit multichannel and silently returns 8-bit,
        which would make ``dtype`` -- what a caller requantizes against -- a lie.
        The channel order is asserted with it, since the reader that can is
        OpenCV's, and that one is BGR."""
        from pgs_recon.utils.images import read_srgb
        import cv2
        import numpy as np
        src = np.array([[[10000, 20000, 60000]]], dtype=np.uint16)
        path = self.tmp / 'x.png'
        cv2.imwrite(str(path), src[..., ::-1])  # written as BGR
        got = read_srgb(path)
        self.assertEqual(got.dtype, np.uint16)
        np.testing.assert_allclose(got.pixels, src / 65535.)

    def test_lab_file_reports_that_it_was_decoded(self):
        """``decoded`` is what stops a caller bit-shifting an already-sRGB image."""
        from pgs_recon.utils.images import CIELAB, read_srgb
        import numpy as np
        samples = encode(lab_row(), CIELAB, np.uint16)
        got = read_srgb(self.write(samples, photometric='cielab'))
        self.assertTrue(got.decoded)
        self.assertEqual(got.dtype, np.uint16)
        self.assertEqual(got.pixels.shape[-1], 3)

    def test_signed_lab_tiff_reads_through(self):
        """The whole path, on the file a signed writer actually produces."""
        from pgs_recon.utils.images import CIELAB, read_srgb
        import numpy as np
        import imageio.v3 as iio
        lab = lab_row()
        path = self.write(encode(lab, CIELAB, np.uint16, True),
                          'cielab', 'signed')
        unsigned = read_srgb(self.write(encode(lab, CIELAB, np.uint16),
                                        'cielab', 'unsigned'))
        got = read_srgb(path)
        self.assertEqual(iio.imread(path).dtype, np.int16)  # SampleFormat = 2
        # ...but no caller can write that back out, and it is not the depth of
        # the decoded image either.
        self.assertEqual(got.dtype, np.uint16)
        self.assertTrue(got.decoded)
        np.testing.assert_array_equal(got.pixels, unsigned.pixels)

    def test_16bit_to_8bit_matches_imagemagick_exactly(self):
        """``convert -depth 8`` is round(v / 257) -- measured, not assumed."""
        from pgs_recon.utils.images import as_truecolor, read_srgb, to_uint8
        import numpy as np
        src = (np.arange(256, dtype=np.uint16).reshape(16, 16) * 257)
        got = to_uint8(as_truecolor(read_srgb(
            self.write(src, photometric='minisblack')).pixels))
        want = np.dstack([np.round(src / 257.).astype(np.uint8)] * 3)
        np.testing.assert_array_equal(got, want)

    def test_8bit_survives_the_round_trip_untouched(self):
        """Truncating instead of rounding here costs a level on most pixels."""
        from pgs_recon.utils.images import read_srgb, to_uint8
        import numpy as np
        src = np.arange(48, dtype=np.uint8).reshape(4, 4, 3)
        np.testing.assert_array_equal(to_uint8(read_srgb(self.write(src)).pixels),
                                      src)

    def test_as_truecolor_shapes(self):
        from pgs_recon.utils.images import as_truecolor
        import numpy as np
        flat = np.zeros((3, 3))
        self.assertEqual(as_truecolor(flat).shape, (3, 3, 3))
        self.assertEqual(as_truecolor(np.zeros((3, 3, 1))).shape, (3, 3, 3))
        self.assertEqual(as_truecolor(np.zeros((3, 3, 3))).shape, (3, 3, 3))
        # alpha is dropped rather than blended into a texture
        self.assertEqual(as_truecolor(np.zeros((3, 3, 4))).shape, (3, 3, 3))

    def test_prepare_8bit_image_writes_three_channel_8bit(self):
        from pgs_recon.utils.images import prepare_8bit_image
        import imageio.v3 as iio
        import numpy as np
        src = (np.arange(256, dtype=np.uint16).reshape(16, 16) * 257)
        out = prepare_8bit_image(self.write(src, photometric='minisblack'),
                                 self.tmp / 'out')
        self.assertEqual(out.suffix, '.jpg')
        got = iio.imread(out)
        self.assertEqual(got.shape, (16, 16, 3))
        self.assertEqual(got.dtype, np.uint8)


if __name__ == '__main__':
    unittest.main()
