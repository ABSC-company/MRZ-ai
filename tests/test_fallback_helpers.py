import unittest

from app.services.fallback import FallbackFieldExtractor, normalize_visual_value


class FakeScalar:
    def __init__(self, value):
        self.value = value

    def item(self):
        return self.value


class FakeBox:
    def __init__(self, values):
        self.values = values

    def tolist(self):
        return self.values


class FallbackHelperTests(unittest.TestCase):
    def test_visual_value_normalization(self):
        self.assertEqual(normalize_visual_value("birth_date", "03.11.2006"), "2006-11-03")
        self.assertEqual(normalize_visual_value("personal_number", "061 103 502 489"), "061103502489")
        self.assertEqual(normalize_visual_value("sex", "Ж"), "F")
        self.assertEqual(normalize_visual_value("surname", "  Осипович "), "ОСИПОВИЧ")

    def test_only_highest_score_is_kept_for_each_class(self):
        extractor = object.__new__(FallbackFieldExtractor)
        extractor.threshold = 0.40
        detections = extractor._best_per_class(
            labels=[FakeScalar(2), FakeScalar(2), FakeScalar(3), FakeScalar(5)],
            boxes=[
                FakeBox([10, 10, 50, 30]),
                FakeBox([12, 11, 52, 31]),
                FakeBox([10, 40, 60, 65]),
                FakeBox([0, 0, 10, 10]),
            ],
            scores=[FakeScalar(0.65), FakeScalar(0.91), FakeScalar(0.80), FakeScalar(0.20)],
            size=(100, 100),
        )
        self.assertEqual([item.class_id for item in detections], [2, 3])
        self.assertAlmostEqual(detections[0].score, 0.91)
        self.assertEqual(detections[0].box, (12, 11, 52, 31))


if __name__ == "__main__":
    unittest.main()
