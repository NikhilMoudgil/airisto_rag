"""Tests for Component 2 (indexer). Run: python -m unittest test_indexer -v"""
import os
import tempfile
import unittest

from embedders import HashingEmbedder
from indexer import build_index, code_variants, index_text, load_index, record_from_dict, record_to_dict, save_index
from schema import load_csv

DATA = os.path.join(os.path.dirname(__file__), "data", "test_records_FAKE.csv")


class IndexerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.records, errors = load_csv(DATA)
        assert not errors
        cls.emb = HashingEmbedder()

    def test_unreviewed_left_out_by_default(self):
        index, skipped = build_index(self.records, self.emb)
        self.assertEqual(skipped, ["fakeair-all-f99"])
        self.assertEqual(index["count"], len(self.records) - 1)

    def test_include_unreviewed_flag(self):
        index, skipped = build_index(self.records, self.emb, include_unreviewed=True)
        self.assertEqual(skipped, [])
        self.assertEqual(index["count"], len(self.records))

    def test_code_variants_cover_typing_styles(self):
        self.assertEqual(code_variants("CH 07"), ["CH 07", "CH07", "CH-07"])
        self.assertEqual(code_variants("F-12"), ["F-12", "F12", "F 12"])

    def test_index_text_has_key_parts(self):
        rec = next(r for r in self.records if r.code == "F-12")
        text = index_text(rec)
        for part in ("FakeAir", "F12", "paani tapak raha hai", "blocked drain pipe"):
            self.assertIn(part, text)

    def test_record_roundtrip(self):
        for r in self.records:
            self.assertEqual(record_from_dict(record_to_dict(r)), r)

    def test_vectors_are_normalised(self):
        index, _ = build_index(self.records, self.emb)
        for e in index["entries"]:
            self.assertAlmostEqual(sum(x * x for x in e["vector"]), 1.0, places=6)
            self.assertEqual(len(e["vector"]), self.emb.dim)

    def test_save_load_and_mismatch(self):
        index, _ = build_index(self.records, self.emb)
        with tempfile.TemporaryDirectory() as d:
            save_index(index, d)
            self.assertEqual(load_index(d, self.emb)["count"], index["count"])
            with self.assertRaises(ValueError):
                load_index(d, HashingEmbedder(dim=256))

    def test_version_mismatch_refused(self):
        index, _ = build_index(self.records, self.emb)
        index["version"] = 999
        with tempfile.TemporaryDirectory() as d:
            save_index(index, d)
            with self.assertRaises(ValueError):
                load_index(d)


if __name__ == "__main__":
    unittest.main()
