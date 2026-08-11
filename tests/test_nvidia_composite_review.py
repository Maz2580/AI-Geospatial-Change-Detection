from __future__ import annotations

import unittest

import cv2
import numpy as np

from review_candidates_nvidia_composite import _comparison_jpeg


class NvidiaCompositeImageTests(unittest.TestCase):
    def test_combines_two_rendered_crops_into_one_labelled_jpeg(self) -> None:
        before = np.full((64, 80, 3), (20, 30, 40), dtype=np.uint8)
        after = np.full((80, 48, 3), (80, 90, 100), dtype=np.uint8)
        ok_before, before_jpeg = cv2.imencode(".jpg", before)
        ok_after, after_jpeg = cv2.imencode(".jpg", after)

        self.assertTrue(ok_before)
        self.assertTrue(ok_after)
        output = _comparison_jpeg(before_jpeg.tobytes(), after_jpeg.tobytes())
        decoded = cv2.imdecode(np.frombuffer(output, dtype=np.uint8), cv2.IMREAD_COLOR)

        self.assertIsNotNone(decoded)
        self.assertGreater(decoded.shape[1], 80)
        self.assertGreater(decoded.shape[0], 80)
        self.assertLess(len(output), 110_000)


if __name__ == "__main__":
    unittest.main()
