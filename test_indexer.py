"""Tests for Component 2. Run: python -m unittest test_indexer -v"""
import os
import tempfile
import unittest

from embedders import HashingEmbedder, cosine
from indexer import (build_index, code_variants, index_text, load_index,
                     main, record_from_dict, save_index)
from schema import load_csv

DATA = os.path.join(os.path.dirname(__file__), "data", "test_records_FAKE.csv")


class IndexerTests(unittest.TestCase):
    def setUp(self):
        self.records, errors = load_csv(DATA)
        self.assertEqual(errors, [])
        self.emb = HashingEmbedder()

    def test_code_variants(self):
        self.assertEqual(code_variants("CH 07"), ["CH 07", "CH07", "CH-07"])
        self.assertEqual(code_variants("E4"), ["E4", "E 4", "E-4"])
        self.assertIn("F 12", code_variants("F-12"))

    def test_index_text_has_all_signals(self):
        rec = next(r for r in self.records if r.code == "T3")
        text = index_text(rec)
        for needle in ("TestCo", "T3", "fan motor fault", "fan chal nahi raha", "faulty capacitor"):
            self.assertIn(needle, text)

    def test_unreviewed_excluded_by_default(self):
        index, skipped = build_index(self.records, self.emb)
        self.assertEqual(skipped, ["fakeair-all-f99"])
        self.assertEqual(index["count"], len(self.records) - 1)
        ids = [e["id"] for e in index["entries"]]
        self.assertNotIn("fakeair-all-f99", ids)

    def test_include_unreviewed_flag(self):
        index, skipped = build_index(self.records, self.emb, include_unreviewed=True)
        self.assertEqual(skipped, [])
        self.assertEqual(index["count"], len(self.records))

    def test_same_code_different_series_both_kept(self):
        index, _ = build_index(self.records, self.emb)
        ids = {e["id"] for e in index["entries"]}
        self.assertIn("testco-x100-t5", ids)
        self.assertIn("testco-z200-t5", ids)

    def test_roundtrip_and_embedder_guard(self):
        index, _ = build_index(self.records, self.emb)
        with tempfile.TemporaryDirectory() as d:
            save_index(index, d)
            loaded = load_index(d, self.emb)
            self.assertEqual(loaded["count"], index["count"])
            rec = record_from_dict(loaded["entries"][0]["record"])
            self.assertEqual(rec.record_id, loaded["entries"][0]["id"])
            with self.assertRaises(ValueError):
                load_index(d, HashingEmbedder(dim=256))

    def test_vectors_normalised_and_deterministic(self):
        v1 = self.emb.embed_query("fan not spinning")
        v2 = HashingEmbedder().embed_query("fan not spinning")
        self.assertEqual(v1, v2)
        self.assertAlmostEqual(cosine(v1, v1), 1.0, places=6)

    def test_similar_text_scores_higher_than_unrelated(self):
        q = self.emb.embed_query("indoor fan motor fault")
        near = self.emb.embed_query("Meaning: Indoor fan motor fault (fictional)")
        far = self.emb.embed_query("Water leak detected blocked drain pipe")
        self.assertGreater(cosine(q, near), cosine(q, far))

    def test_cli_refuses_bad_csv(self):
        with tempfile.TemporaryDirectory() as d:
            bad = os.path.join(d, "bad.csv")
            with open(bad, "w") as f:
                f.write("brand,model_series,code,meaning,likely_causes,user_checks,verdict,track,"
                        "warnings,typical_parts,symptoms,source,reviewed_by,last_reviewed\n"
                        "A,,E1,m,c,,urgent,repair,,,,src,,\n")
            self.assertEqual(main([bad, "--out", os.path.join(d, "idx")]), 1)
            self.assertFalse(os.path.exists(os.path.join(d, "idx", "index.json")))

    def test_cli_builds_index(self):
        with tempfile.TemporaryDirectory() as d:
            self.assertEqual(main([DATA, "--out", d]), 0)
            self.assertTrue(os.path.exists(os.path.join(d, "index.json")))


if __name__ == "__main__":
    unittest.main()
