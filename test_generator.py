"""Tests for Component 4 (generator) and the pipeline. Run: python -m unittest test_generator -v"""
import os
import unittest
from urllib.parse import parse_qs, urlparse

from embedders import HashingEmbedder
from generator import WRAPPERS, build_response, detect_language, wrapper_ok
from indexer import build_index
from pipeline import answer
from retriever import Retriever
from schema import load_csv

DATA = os.path.join(os.path.dirname(__file__), "data", "test_records_FAKE.csv")
CFG = {"whatsapp_number": "919876543210", "discount_code": "AIR50"}


class GeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        records, _ = load_csv(DATA)
        emb = HashingEmbedder()
        index, _ = build_index(records, emb)
        cls.ret = Retriever(index, emb)

    def ask(self, q, state=None, llm=None, **kw):
        return answer(q, self.ret, state, llm=llm, **CFG, **kw)

    # ----- language -----
    def test_detect_language(self):
        self.assertEqual(detect_language("LG AC CH 07 not cooling"), "en")
        self.assertEqual(detect_language("mera ac thanda nahi kar raha"), "hinglish")
        self.assertEqual(detect_language("मेरा एसी चालू नहीं हो रहा"), "hi")

    # ----- card fields come from the record -----
    def test_exact_card(self):
        r, _ = self.ask("TestCo T3")
        self.assertEqual((r["match_type"], r["verdict"], r["code"]), ("exact", "technician_required", "T3"))
        self.assertEqual(r["meaning"], "[FAKE] Indoor fan motor fault (fictional)")
        self.assertEqual(r["source"], "FAKE TEST DATA")
        self.assertIn("Do not open the unit", r["warnings"])
        self.assertTrue(r["book_url"].startswith("https://wa.me/919876543210"))

    def test_self_fix_has_checks_no_booking_then_escalates(self):
        r, st = self.ask("TestCo T1")
        self.assertEqual(r["verdict"], "self_fix")
        self.assertIsNone(r["book_url"])
        self.assertTrue(r["user_checks"])
        r2, _ = self.ask("TestCo T1 still showing", st)
        self.assertEqual(r2["verdict"], "technician_required")
        self.assertIsNotNone(r2["book_url"])
        self.assertEqual(r2["wrapper"], WRAPPERS["en"]["escalated"])

    def test_urgent_warning_first(self):
        r, _ = self.ask("TestCo T4")
        self.assertEqual(r["verdict"], "urgent")
        self.assertTrue(r["warnings"][0].startswith("Switch the AC off"))
        self.assertIn("URGENT", parse_qs(urlparse(r["book_url"]).query)["text"][0])

    def test_hinglish_wrapper_template(self):
        r, _ = self.ask("mera TestCo T3 error aa raha hai")
        self.assertEqual(r["lang"], "hinglish")
        self.assertEqual(r["wrapper"], WRAPPERS["hinglish"]["technician_required"])

    # ----- clarification / no match -----
    def test_ambiguous_asks_and_gives_no_diagnosis(self):
        r, _ = self.ask("T3")
        self.assertEqual((r["needs"], r["options"], r["verdict"], r["meaning"]), ("brand", ["TestCo"], None, None))
        r, _ = self.ask("TestCo T5")
        self.assertEqual((r["needs"], r["options"]), ("model_series", ["X100", "Z200"]))

    def test_unknown_code_is_honest(self):
        r, _ = self.ask("TestCo T9 fan not spinning")
        self.assertEqual((r["match_type"], r["meaning"], r["verdict"]), ("none", None, None))
        self.assertEqual(r["wrapper"], WRAPPERS["en"]["none"])
        self.assertTrue(r["user_checks"])
        self.assertIn("T9", parse_qs(urlparse(r["book_url"]).query)["text"][0])

    def test_unknown_code_with_smoke_is_urgent(self):
        r, _ = self.ask("TestCo T9 and burning smell")
        self.assertEqual(r["verdict"], "urgent")
        self.assertEqual(r["wrapper"], WRAPPERS["en"]["none_urgent"])

    def test_safety_words_with_ambiguous_code_still_warn(self):
        r, _ = self.ask("T3 burning smell")
        self.assertEqual(r["needs"], "brand")
        self.assertTrue(r["warnings"])

    def test_empty_query(self):
        r, _ = self.ask("   ")
        self.assertEqual(r["wrapper"], WRAPPERS["en"]["empty"])
        self.assertIsNone(r["book_url"])

    def test_semantic_confident_answers(self):
        r, _ = self.ask("water dripping from indoor unit", brand="FakeAir")
        self.assertEqual((r["match_type"], r["code"]), ("semantic", "F-12"))

    # ----- the LLM is only allowed to write the wrapper -----
    def test_good_llm_text_is_used(self):
        r, _ = self.ask("TestCo T3", llm=lambda s, u: "Sorry about the trouble. A technician can sort this out quickly.")
        self.assertTrue(r["wrapper"].startswith("Sorry about"))

    def test_unsafe_llm_text_is_replaced(self):
        for bad in ("Just open the unit and check the capacitor.", "Top up the refrigerant yourself.",
                    "Check the wiring behind the panel.", "x" * 500, "```json {}```"):
            r, _ = self.ask("TestCo T3", llm=lambda s, u, b=bad: b)
            self.assertEqual(r["wrapper"], WRAPPERS["en"]["technician_required"], bad[:30])

    def test_llm_cannot_change_card_fields(self):
        r, _ = self.ask("TestCo T3", llm=lambda s, u: "Everything is fine, no technician needed.")
        self.assertEqual(r["verdict"], "technician_required")
        self.assertIsNotNone(r["book_url"])

    def test_llm_error_falls_back(self):
        def boom(s, u):
            raise RuntimeError("network down")
        r, _ = self.ask("TestCo T3", llm=boom)
        self.assertEqual(r["wrapper"], WRAPPERS["en"]["technician_required"])

    def test_wrapper_ok(self):
        self.assertTrue(wrapper_ok("A technician should take a look."))
        self.assertFalse(wrapper_ok(""))

    def test_response_has_contract_keys(self):
        r, _ = self.ask("TestCo T3")
        for k in ("brand", "code", "meaning", "user_checks", "verdict", "warnings", "source", "wrapper",
                  "book_url", "match_type"):
            self.assertIn(k, r)


if __name__ == "__main__":
    unittest.main()
