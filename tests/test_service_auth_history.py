import sys
from pathlib import Path
from unittest import TestCase

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from service.auth import TokenBucket, check_auth, client_id_for
from service.history import Conversation, HistoryStore


class HistoryTests(TestCase):
    def test_turn_limit_keeps_newest(self):
        conv = Conversation("s", max_turns=2, max_chars=10000)
        for i in range(6):
            conv.add("user", f"msg {i}")
        self.assertEqual(len(conv.messages), 4)  # 2 turns = 4 messages
        self.assertEqual(conv.messages[-1]["content"], "msg 5")
        self.assertGreater(conv.dropped_turns, 0)

    def test_char_budget_trims_and_reports(self):
        conv = Conversation("s", max_turns=20, max_chars=50)
        conv.add("user", "x" * 40)
        conv.add("user", "y" * 40)
        self.assertLessEqual(conv.stats()["chars"], 50)
        self.assertGreater(conv.dropped_chars, 0)

    def test_system_prompt_survives_trimming(self):
        conv = Conversation("s", max_turns=1, max_chars=10000)
        conv.add("system", "be helpful")
        for i in range(4):
            conv.add("user", f"m{i}")
        self.assertEqual(conv.messages[0], {"role": "system", "content": "be helpful"})

    def test_store_is_per_session(self):
        store = HistoryStore()
        store.get("a").add("user", "hello a")
        self.assertEqual(store.get("b").messages, [])
        self.assertEqual(store.session_count(), 2)
        self.assertTrue(store.drop("a"))
        self.assertFalse(store.drop("a"))


class AuthRateLimitTests(TestCase):
    def test_auth_disabled_without_token(self):
        self.assertTrue(check_auth(None, ""))

    def test_auth_requires_exact_bearer_token(self):
        self.assertTrue(check_auth("Bearer abc", "abc"))
        self.assertFalse(check_auth("Bearer wrong", "abc"))
        self.assertFalse(check_auth(None, "abc"))
        self.assertFalse(check_auth("Basic abc", "abc"))

    def test_client_id_never_contains_the_token(self):
        key = client_id_for(True, "Bearer abc", "127.0.0.1")
        self.assertNotIn("abc", key)

    def test_token_bucket_burst_then_refill(self):
        bucket = TokenBucket(per_min=60, burst=2)
        self.assertTrue(bucket.allow("c", now=1000.0)[0])
        self.assertTrue(bucket.allow("c", now=1000.0)[0])
        allowed, retry = bucket.allow("c", now=1000.0)
        self.assertFalse(allowed)
        self.assertGreater(retry, 0)
        # 60s later one token has refilled.
        self.assertTrue(bucket.allow("c", now=1061.0)[0])

    def test_buckets_are_per_client(self):
        bucket = TokenBucket(per_min=60, burst=1)
        self.assertTrue(bucket.allow("a", now=1000.0)[0])
        self.assertTrue(bucket.allow("b", now=1000.0)[0])
