"""Tests for Component 5 (triage + booking). Run: python -m unittest test_triage -v"""
import os
import unittest
from urllib.parse import parse_qs, urlparse

from schema import load_csv
from triage import (SWITCH_OFF_WARNING, ConversationState, booking_url, booking_url_no_match, decide,
                    decide_no_match, fix_failed_in, safety_concern_in)

DATA = os.path.join(os.path.dirname(__file__), "data", "test_records_FAKE.csv")
RECS = {r.code: r for r in load_csv(DATA)[0]}
NUM, DISC = "+91 98765-43210", "AIR50"


def msg(url):
    return parse_qs(urlparse(url).query)["text"][0]


class TriageTests(unittest.TestCase):
    def test_self_fix_no_booking(self):
        d = decide(RECS["T1"])
        self.assertEqual((d.verdict, d.track, d.show_booking, d.escalated), ("self_fix", "none", False, False))
        self.assertIsNone(booking_url(RECS["T1"], d, whatsapp_number=NUM))

    def test_failed_fix_escalates_by_rule(self):
        d = decide(RECS["T1"], ConversationState(fix_failed=True))
        self.assertEqual((d.verdict, d.track, d.show_booking, d.escalated), ("technician_required", "repair", True, True))

    def test_other_verdicts_unchanged_and_bookable(self):
        for code, verdict, track in (("T2", "needs_service", "service"), ("T3", "technician_required", "repair")):
            d = decide(RECS[code], ConversationState(fix_failed=True))
            self.assertEqual((d.verdict, d.track, d.show_booking, d.escalated), (verdict, track, True, False))

    def test_urgent_warning_first(self):
        d = decide(RECS["T4"])
        self.assertEqual(d.warnings[0], SWITCH_OFF_WARNING)
        self.assertTrue(d.switch_off_first)
        self.assertGreater(len(d.warnings), 1)  # record's own warnings kept

    def test_fix_failed_phrases(self):
        for t in ("still showing the same code", "it didn't work", "didnt work", "abhi bhi error aa raha hai",
                  "theek nahi hua", "error came back", "I already tried that"):
            self.assertTrue(fix_failed_in(t), t)
        for t in ("TestCo T1", "ac not cooling", "thanks it works now"):
            self.assertFalse(fix_failed_in(t), t)

    def test_state_is_sticky(self):
        s1 = ConversationState.from_message("still showing T1")
        s2 = ConversationState.from_message("ok what now", previous=s1)
        self.assertTrue(s2.fix_failed)

    def test_booking_url_contents(self):
        d = decide(RECS["T3"])
        url = booking_url(RECS["T3"], d, ConversationState(model="Foo 1.5T"), NUM, DISC)
        self.assertTrue(url.startswith("https://wa.me/919876543210?text="))
        text = msg(url)
        for part in ("TestCo", "T3", "Foo 1.5T", "Repair", DISC):
            self.assertIn(part, text)

    def test_booking_url_urgent_and_escalated_notes(self):
        self.assertIn("URGENT", msg(booking_url(RECS["T4"], decide(RECS["T4"]), whatsapp_number=NUM, discount_code="")))
        d = decide(RECS["T1"], ConversationState(fix_failed=True))
        self.assertIn("still there", msg(booking_url(RECS["T1"], d, whatsapp_number=NUM, discount_code="")))

    def test_booking_url_needs_number(self):
        os.environ.pop("WHATSAPP_NUMBER", None)
        self.assertIsNone(booking_url(RECS["T3"], decide(RECS["T3"])))

    def test_no_match(self):
        d = decide_no_match("testco t9 not cooling")
        self.assertIsNone(d.verdict)
        self.assertTrue(d.show_booking)
        url = booking_url_no_match(d, brand="TestCo", code_typed="T9", text="not cooling", whatsapp_number=NUM, discount_code="")
        self.assertIn("T9", msg(url))

    def test_no_match_safety_words_are_urgent(self):
        for t in ("ac se dhuan nikal raha hai", "burning smell from ac", "sparks from outdoor unit"):
            self.assertTrue(safety_concern_in(t), t)
            d = decide_no_match(t)
            self.assertEqual(d.verdict, "urgent")
            self.assertEqual(d.warnings, [SWITCH_OFF_WARNING])


if __name__ == "__main__":
    unittest.main()
