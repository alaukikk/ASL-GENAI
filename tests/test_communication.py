"""Unit tests for communication.py. No camera, display, network or Groq key needed.

    python -m unittest discover -s tests -v
"""

import sys
import unittest
from pathlib import Path
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import communication as cm  # noqa: E402


class FakeClient:
    """Stands in for groq.Groq and records every request."""

    def __init__(self, *replies, fail=None):
        self.replies = list(replies)
        self.fail = fail
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        if self.fail is not None:
            raise self.fail
        reply = self.replies.pop(0)
        message = SimpleNamespace(content=reply)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def service(*replies, fail=None):
    client = FakeClient(*replies, fail=fail)
    return cm.GroqService(api_key="test", client=client), client


class Named(Exception):
    """Builds exceptions whose class name matches the Groq SDK's, without the SDK."""


def sdk_error(name, status=None):
    error = type(name, (Exception,), {})("boom")
    if status is not None:
        error.status_code = status
    return error


class LanguageTests(unittest.TestCase):
    def test_three_languages(self):
        self.assertEqual([l.name for l in cm.LANGUAGES.values()], ["English", "Hindi", "Marathi"])

    def test_unknown_code_falls_back_to_english(self):
        self.assertEqual(cm.get_language("xx").code, "en")

    def test_correction_prompt_targets_selected_language(self):
        self.assertIn("natural, correct English", cm.correction_prompt(cm.get_language("en")))
        hindi = cm.correction_prompt(cm.get_language("hi"))
        self.assertIn("Hindi", hindi)
        self.assertIn("Devanagari", hindi)
        marathi = cm.correction_prompt(cm.get_language("mr"))
        self.assertIn("Marathi", marathi)
        self.assertNotIn("Hindi", marathi)

    def test_correction_prompt_forbids_inventing(self):
        prompt = cm.correction_prompt(cm.get_language("en"))
        self.assertIn("Never add facts", prompt)
        self.assertIn("UNCLEAR", prompt)

    def test_reply_prompt_pins_language(self):
        for code, name in (("en", "English"), ("hi", "Hindi"), ("mr", "Marathi")):
            self.assertIn(f"Always reply in {name}", cm.reply_prompt(cm.get_language(code)))


class GroqServiceTests(unittest.TestCase):
    def test_correct_sends_wrapped_input_and_selected_language(self):
        svc, client = service("Hello, how are you?")
        self.assertEqual(svc.correct("HELLOHOW ARE YOU", "en"), "Hello, how are you?")
        call = client.calls[0]
        self.assertEqual(call["model"], cm.MODEL)
        self.assertEqual(call["messages"][1]["content"], "<signed>HELLOHOW ARE YOU</signed>")
        self.assertIn("English", call["messages"][0]["content"])

    def test_hindi_correction_uses_hindi_prompt(self):
        svc, client = service("नमस्ते, आप कैसे हैं?")
        self.assertEqual(svc.correct("HELLO HOW ARE YOU", "hi"), "नमस्ते, आप कैसे हैं?")
        self.assertIn("Hindi", client.calls[0]["messages"][0]["content"])

    def test_unclear_is_detected(self):
        for text in ("UNCLEAR", "unclear.", '"UNCLEAR"'):
            svc, _ = service(text)
            with self.assertRaises(cm.UnclearMessage):
                svc.correct("QXZ", "en")

    def test_wrapping_is_stripped(self):
        svc, _ = service('"I need water."')
        self.assertEqual(svc.correct("I NEED WATER", "en"), "I need water.")
        self.assertEqual(cm.clean_model_text("<think>hm</think>Hi"), "Hi")
        self.assertEqual(cm.clean_model_text("```\nHi\n```"), "Hi")

    def test_reply_uses_language_history_and_message(self):
        svc, client = service("Marathi reply")
        history = [{"role": "user", "content": "a"}, {"role": "assistant", "content": "b"}]
        self.assertEqual(svc.reply("hello", "mr", history), "Marathi reply")
        messages = client.calls[0]["messages"]
        self.assertEqual([m["role"] for m in messages], ["system", "user", "assistant", "user"])
        self.assertIn("Always reply in Marathi", messages[0]["content"])
        self.assertEqual(messages[-1]["content"], "hello")

    def test_empty_response_is_an_error(self):
        svc, _ = service("   ")
        with self.assertRaises(cm.AIError):
            svc.reply("hi", "en")

    def test_truncated_empty_response_says_so(self):
        client = FakeClient("")
        original = client._create

        def truncated(**kw):
            response = original(**kw)
            response.choices[0].finish_reason = "length"
            return response

        client.chat.completions.create = truncated
        with self.assertRaises(cm.AIError) as raised:
            cm.GroqService(api_key="k", client=client).reply("hi", "en")
        self.assertIn("ran out of space", str(raised.exception))

    def test_missing_key(self):
        svc = cm.GroqService(api_key="")
        self.assertFalse(svc.configured)
        with self.assertRaises(cm.AIError) as raised:
            svc.reply("hi", "en")
        self.assertIn("GROQ_API_KEY", str(raised.exception))

    def test_key_read_from_environment(self):
        import os
        old = os.environ.get("GROQ_API_KEY")
        os.environ["GROQ_API_KEY"] = "  abc  "
        try:
            self.assertEqual(cm.GroqService().api_key, "abc")
        finally:
            if old is None:
                del os.environ["GROQ_API_KEY"]
            else:
                os.environ["GROQ_API_KEY"] = old

    def test_api_failures_become_friendly_messages(self):
        cases = {
            "AuthenticationError": "API key was rejected",
            "RateLimitError": "rate-limiting",
            "APITimeoutError": "too long",
            "APIConnectionError": "Could not reach",
        }
        for name, expected in cases.items():
            svc, _ = service(fail=sdk_error(name))
            with self.assertRaises(cm.AIError) as raised:
                svc.reply("hi", "en")
            self.assertIn(expected, str(raised.exception), name)
        svc, _ = service(fail=sdk_error("InternalServerError", 503))
        with self.assertRaises(cm.AIError) as raised:
            svc.reply("hi", "en")
        self.assertIn("having problems", str(raised.exception))

    def test_history_trim_never_starts_with_assistant(self):
        history = [{"role": "assistant", "content": "x"}, {"role": "user", "content": "y"}]
        self.assertEqual(cm.trim_history(history)[0]["role"], "user")
        long = []
        for i in range(40):
            long += [{"role": "user", "content": str(i)}, {"role": "assistant", "content": str(i)}]
        trimmed = cm.trim_history(long)
        self.assertLessEqual(len(trimmed), cm.HISTORY_LIMIT)
        self.assertEqual(trimmed[0]["role"], "user")


class RunJobTests(unittest.TestCase):
    def test_signed_goes_through_correction_then_reply(self):
        svc, client = service("Hello.", "Hi there!")
        job = cm.Job(1, "HELLO", "en", "signed")
        seen = []
        reply = cm.run_job(svc, job, [], on_corrected=lambda j: seen.append(j.corrected))
        self.assertEqual((job.corrected, reply, seen), ("Hello.", "Hi there!", ["Hello."]))
        self.assertEqual(len(client.calls), 2)
        # The assistant receives the corrected sentence as an ordinary user message.
        self.assertEqual(client.calls[1]["messages"][-1], {"role": "user", "content": "Hello."})

    def test_typed_skips_correction(self):
        svc, client = service("Sure.")
        job = cm.Job(1, "what time is it", "en", "typed")
        cm.run_job(svc, job, [])
        self.assertEqual(len(client.calls), 1)

    def test_retry_after_reply_failure_reuses_correction(self):
        svc, client = service("Reply")
        job = cm.Job(1, "HELLO", "en", "signed", corrected="Hello.")
        cm.run_job(svc, job, [])
        self.assertEqual(len(client.calls), 1)       # no second correction call

    def test_unclear_never_reaches_assistant(self):
        svc, client = service("UNCLEAR")
        with self.assertRaises(cm.UnclearMessage):
            cm.run_job(svc, cm.Job(1, "QXZ", "en", "signed"), [])
        self.assertEqual(len(client.calls), 1)


class JobQueueTests(unittest.TestCase):
    def setUp(self):
        self.now = 100.0
        self.queue = cm.JobQueue(clock=lambda: self.now)

    def test_empty_and_symbol_only_refused(self):
        for text in ("", "   ", ". , !"):
            self.assertEqual(self.queue.submit(text, "en"), (None, "empty"))

    def test_duplicate_within_window_refused_then_allowed(self):
        job, _ = self.queue.submit("HELLO", "en")
        self.assertIsNotNone(job)
        self.assertEqual(self.queue.submit(" hello"[1:].upper(), "en"), (None, "duplicate"))
        self.now += 2.0
        self.assertIsNotNone(self.queue.submit("HELLO", "en")[0])

    def test_whitespace_normalised(self):
        job, _ = self.queue.submit("  HOW   ARE  YOU ", "en")
        self.assertEqual(job.raw, "HOW ARE YOU")

    def test_one_job_at_a_time_in_order(self):
        a, _ = self.queue.submit("ONE", "en")
        b, _ = self.queue.submit("TWO", "hi")
        self.assertIs(self.queue.next(), a)
        self.assertIsNone(self.queue.next())         # a is still active
        self.queue.finish(a)
        self.assertIs(self.queue.next(), b)
        self.assertEqual(b.language, "hi")

    def test_backlog_limit(self):
        for i in range(cm.JobQueue.MAX_PENDING):
            self.assertIsNotNone(self.queue.submit(f"M{i}", "en")[0])
        self.assertEqual(self.queue.submit("OVER", "en"), (None, "busy"))

    def test_retry_only_failed(self):
        job, _ = self.queue.submit("HELLO", "en")
        self.queue.next()
        self.assertFalse(self.queue.retry(job))
        job.state = "failed"
        self.queue.finish(job)
        self.assertTrue(self.queue.retry(job))
        self.assertIs(self.queue.next(), job)

    def test_state_on_start(self):
        signed, _ = self.queue.submit("A", "en", "signed")
        self.assertEqual(self.queue.next().state, "refining")
        self.queue.finish(signed)
        typed, _ = self.queue.submit("typed text", "en", "typed")
        self.assertEqual(self.queue.next().state, "replying")


class SegmenterTests(unittest.TestCase):
    def test_sends_after_pause_only_once_text_exists(self):
        seg = cm.MessageSegmenter(word_gap=1.2, send_delay=3.0)
        seg.reset(0.0)
        self.assertIsNone(seg.update("", True, 10.0))
        self.assertIsNone(seg.update("", True, 20.0))          # nothing to send
        self.assertIsNone(seg.update("HI", True, 21.0))
        self.assertIsNone(seg.update("HI", True, 23.9))
        self.assertEqual(seg.update("HI", True, 24.0), "send")

    def test_new_letter_restarts_the_clock(self):
        seg = cm.MessageSegmenter(send_delay=3.0)
        seg.reset(0.0)
        seg.update("H", True, 1.0)
        seg.update("HI", True, 3.5)
        self.assertIsNone(seg.update("HI", True, 6.0))
        self.assertEqual(seg.update("HI", True, 6.5), "send")

    def test_space_inserted_once_when_hand_leaves(self):
        seg = cm.MessageSegmenter(word_gap=1.2, send_delay=5.0)
        seg.reset(0.0)
        seg.update("HELLO", True, 1.0)
        self.assertIsNone(seg.update("HELLO", False, 1.5))
        self.assertIsNone(seg.update("HELLO", False, 2.6))
        self.assertEqual(seg.update("HELLO", False, 2.8), "space")
        self.assertIsNone(seg.update("HELLO", False, 2.9))      # text not yet updated
        self.assertIsNone(seg.update("HELLO ", False, 3.0))     # and not again afterwards

    def test_no_space_while_hand_is_visible(self):
        seg = cm.MessageSegmenter(word_gap=1.2, send_delay=5.0)
        seg.reset(0.0)
        seg.update("HELLO", True, 1.0)
        self.assertIsNone(seg.update("HELLO", True, 4.0))

    def test_inserted_space_does_not_delay_sending(self):
        seg = cm.MessageSegmenter(word_gap=1.0, send_delay=3.0)
        seg.reset(0.0)
        seg.update("HI", True, 0.0)
        seg.update("HI", False, 0.5)
        self.assertEqual(seg.update("HI", False, 1.6), "space")
        self.assertEqual(seg.update("HI ", False, 3.0), "send")

    def test_suspended_holds_everything(self):
        seg = cm.MessageSegmenter(send_delay=3.0)
        seg.reset(0.0)
        seg.update("HI", True, 0.0)
        self.assertIsNone(seg.update("HI", True, 10.0, suspended=True))
        self.assertIsNone(seg.update("HI", True, 12.0))         # clock restarted at 10
        self.assertEqual(seg.update("HI", True, 13.0), "send")

    def test_punctuation_only_never_sends(self):
        seg = cm.MessageSegmenter(send_delay=1.0)
        seg.reset(0.0)
        self.assertIsNone(seg.update(". ,", True, 50.0))
        self.assertIsNone(seg.remaining(50.0))

    def test_remaining(self):
        seg = cm.MessageSegmenter(send_delay=3.0)
        seg.reset(0.0)
        seg.update("HI", True, 1.0)
        self.assertAlmostEqual(seg.remaining(2.0), 2.0)
        self.assertEqual(seg.remaining(9.0), 0.0)


def voice(id_, name, languages=()):
    return SimpleNamespace(id=id_, name=name, languages=list(languages))


WINDOWS_ALL = [
    voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_EN-US_DAVID_11.0",
          "Microsoft David Desktop - English (United States)"),
    voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_HI-IN_HEMANT_11.0",
          "Microsoft Hemant - Hindi (India)"),
    voice(r"HKEY_LOCAL_MACHINE\SOFTWARE\Microsoft\Speech\Voices\Tokens\TTS_MS_MR-IN_AAROHI_11.0",
          "Microsoft Aarohi - Marathi (India)"),
]


class VoiceTests(unittest.TestCase):
    def test_each_language_gets_its_own_voice(self):
        for code, expected in (("en", "David"), ("hi", "Hemant"), ("mr", "Aarohi")):
            chosen, matched = cm.select_voice(WINDOWS_ALL, code)
            self.assertIn(expected, chosen.name)
            self.assertEqual(matched, code)

    def test_marathi_falls_back_to_hindi_and_says_so(self):
        chosen, matched = cm.select_voice(WINDOWS_ALL[:2], "mr")
        self.assertIn("Hemant", chosen.name)
        self.assertEqual(matched, "hi")

    def test_no_match_returns_none(self):
        self.assertEqual(cm.select_voice(WINDOWS_ALL[:1], "hi"), (None, None))
        self.assertEqual(cm.select_voice([], "mr"), (None, None))

    def test_english_is_not_matched_by_substrings(self):
        # "hi" must not match inside words such as "Chinese" or "Microsoft Hazel".
        voices = [voice("TTS_MS_ZH-CN_HUIHUI", "Microsoft Huihui - Chinese (Simplified)")]
        self.assertEqual(cm.select_voice(voices, "hi"), (None, None))

    def test_espeak_style_voices(self):
        voices = [voice("english", "English", [b"\x05en-gb"]), voice("hindi", "hindi", [b"\x05hi"]),
                  voice("marathi", "marathi", [b"\x05mr"])]
        self.assertEqual(cm.select_voice(voices, "mr")[0].id, "marathi")
        self.assertEqual(cm.select_voice(voices, "hi")[0].id, "hindi")

    def test_speakable_removes_markdown(self):
        self.assertEqual(cm.speakable("**Hello** there\n- one\n- two `x`"), "Hello there one two x")
        self.assertEqual(cm.speakable("See [docs](http://x.y) now"), "See docs now")
        self.assertEqual(cm.speakable("नमस्ते। आप कैसे हैं?"), "नमस्ते। आप कैसे हैं?")


if __name__ == "__main__":
    unittest.main()
