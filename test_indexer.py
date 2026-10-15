"""Tests for Component 3. Run: python -m unittest test_retriever -v"""
import os
import tempfile
import unittest

from embedders import HashingEmbedder
from indexer import build_index, save_index
from retriever import Retriever
from schema import load_csv

DATA = os.path.join(os.path.dirname(__file__), "data", "test_records_FAKE.csv")


class RetrieverTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records, errors = load_csv(DATA)
        assert not errors
        cls.emb = HashingEmbedder()
        index, _ = build_index(records, cls.emb)  # unreviewed F-99 is excluded
        cls.ret = Retriever(index, cls.emb)

    # ----- exact path -----
    def test_exact_brand_and_code(self):
        res = self.ret.retrieve("TestCo T3")
        self.assertEqual(res.match_type, "exact")
        self.assertEqual(res.top.record.record_id, "testco-all-t3")
        self.assertEqual(res.top.score, 1.0)

    def test_code_typed_many_ways(self):
        for q in ("testco t 3", "TESTCO T-3", "my testco ac shows t3 error", "testco error t3 not cooling"):
            res = self.ret.retrieve(q)
            self.assertEqual(res.match_type, "exact", q)
            self.assertEqual(res.top.record.code, "T3", q)

    def test_hyphenated_code_with_space(self):
        res = self.ret.retrieve("fakeair f 12 showing")
        self.assertEqual(res.match_type, "exact")
        self.assertEqual(res.top.record.code, "F-12")

    def test_explicit_ui_choices_win(self):
        res = self.ret.retrieve("", brand="TestCo", code="t 4")
        self.assertEqual(res.match_type, "exact")
        self.assertEqual(res.top.record.verdict, "urgent")

    # ----- clarification path -----
    def test_code_without_brand_asks_for_brand(self):
        res = self.ret.retrieve("T3")
        self.assertEqual(res.match_type, "ambiguous")
        self.assertEqual(res.clarification, "brand")
        self.assertEqual(res.options, ["TestCo"])
        self.assertEqual(res.matches, [])  # never answers before the brand is known

    def test_same_code_two_series_asks_for_model(self):
        res = self.ret.retrieve("TestCo T5")
        self.assertEqual(res.match_type, "ambiguous")
        self.assertEqual(res.clarification, "model_series")
        self.assertEqual(res.options, ["X100", "Z200"])

    def test_model_series_resolves_it(self):
        self.assertEqual(self.ret.retrieve("TestCo X100 T5").top.record.model_series, "X100")
        res = self.ret.retrieve("TestCo T5", model_series="Z200")
        self.assertEqual(res.match_type, "exact")
        self.assertEqual(res.top.record.model_series, "Z200")

    # ----- never guess -----
    def test_unknown_code_for_known_brand_is_not_guessed(self):
        res = self.ret.retrieve("TestCo T9 fan not spinning")
        self.assertEqual(res.match_type, "none")
        self.assertEqual(res.reason, "unknown_code")

    def test_typed_unknown_code_variants(self):
        for q in ("TestCo CH 99", "testco shows e99 error", "testco error code t9 and fan slow"):
            res = self.ret.retrieve(q)
            self.assertEqual((res.match_type, res.reason), ("none", "unknown_code"), q)

    def test_gas_names_and_model_numbers_are_not_codes(self):
        # R32 is a refrigerant; a lowercase model number far from any code word is not a code attempt
        res = self.ret.retrieve("testco r32 gas leak water dripping from indoor unit", brand="TestCo")
        self.assertNotEqual(res.reason, "unknown_code")
        res = self.ret.retrieve("my testco ac model ar18 has indoor fan not spinning today")
        self.assertNotEqual(res.reason, "unknown_code")

    def test_unknown_brand_param(self):
        res = self.ret.retrieve("T3", brand="NoSuchBrand")
        self.assertEqual((res.match_type, res.reason), ("none", "unknown_brand"))

    def test_unreviewed_record_not_retrievable(self):
        self.assertEqual(self.ret.retrieve("FakeAir F-99").match_type, "none")

    def test_empty_query(self):
        self.assertEqual(self.ret.retrieve("   ").reason, "empty_query")

    def test_gibberish_returns_none(self):
        res = self.ret.retrieve("qwxz vbnm plokij")
        self.assertEqual(res.match_type, "none")
        self.assertEqual(res.reason, "no_semantic_match")

    # ----- semantic path -----
    def test_semantic_symptom_with_brand(self):
        res = self.ret.retrieve("water dripping from indoor unit", brand="FakeAir")
        self.assertEqual(res.match_type, "semantic")
        self.assertEqual(res.top.record.code, "F-12")

    def test_hinglish_symptom_phrase(self):
        res = self.ret.retrieve("paani tapak raha hai", brand="FakeAir")
        self.assertEqual(res.top.record.code, "F-12")
        res = self.ret.retrieve("fan chal nahi raha", brand="TestCo")
        self.assertEqual(res.top.record.code, "T3")

    def test_semantic_respects_brand_filter(self):
        res = self.ret.retrieve("water dripping from indoor unit", brand="TestCo")
        for m in res.matches:
            self.assertEqual(m.record.brand, "TestCo")

    # ----- parsing guards -----
    def test_bare_number_is_not_a_code(self):
        r = Retriever.__new__(Retriever)
        r.embedder = self.emb
        r.entries = self.ret.entries
        r._build_vocab()
        r.codes.add("4")
        self.assertIsNone(r.parse_query("testco 1.5 ton ac with 4 star rating not cooling at all")["code"])
        self.assertEqual(r.parse_query("testco error 4")["code"], "4")

    def test_lowercase_short_word_is_not_a_code(self):
        r = Retriever.__new__(Retriever)
        r.embedder = self.emb
        r.entries = self.ret.entries
        r._build_vocab()
        r.codes.add("E")
        self.assertIsNone(r.parse_query("testco ac is e blinking and weak")["code"])
        self.assertEqual(r.parse_query("testco ac shows E")["code"], "E")

    def test_embedder_mismatch_is_refused(self):
        records, _ = load_csv(DATA)
        index, _ = build_index(records, self.emb)
        with tempfile.TemporaryDirectory() as d:
            save_index(index, d)
            with self.assertRaises(ValueError):
                Retriever.from_dir(d, HashingEmbedder(dim=256))


if __name__ == "__main__":
    unittest.main()
