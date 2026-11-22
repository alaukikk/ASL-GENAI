"""End-to-end tests for asl_studio.py that need no Qt, webcam, MediaPipe, network or API key.

What is real here: asl_studio.py itself (camera loop, window logic, transcript rendering),
the ONNX model and ONNX Runtime, OpenCV, preprocessing, SpellingBuffer, MotionTracker and
communication.py.

What is faked: PySide6 (tests/qt_stub.py; nothing is drawn), MediaPipe's hand detector
(a scripted sequence of landmark sets in tests/fixtures/synthetic_poses.json, which the
real model classifies as M and N), the Groq client, the speech engine and the clock.

    python -m unittest discover -s tests -v
"""

import json
import sys
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

TESTS = Path(__file__).resolve().parent
ROOT = TESTS.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(TESTS))

import qt_stub  # noqa: E402

qt_stub.install()

# realtime_translator imports genai_assistant, which imports tkinter, pyttsx3 and groq at
# module level. Provide light stand-ins for whichever of them this machine lacks.
for name in ("tkinter", "tkinter.ttk", "tkinter.messagebox"):
    try:
        __import__(name)
    except ImportError:
        sys.modules[name] = types.ModuleType(name)
        if name != "tkinter":
            setattr(sys.modules["tkinter"], name.split(".")[1], sys.modules[name])
try:
    import groq  # noqa: F401
except ImportError:
    sys.modules["groq"] = types.SimpleNamespace(Groq=object)
try:
    import pyttsx3  # noqa: F401
except ImportError:
    sys.modules["pyttsx3"] = types.ModuleType("pyttsx3")


class FakeEngine:
    def __init__(self, log, voices):
        self.log, self.voices = log, voices

    def getProperty(self, name):
        return self.voices if name == "voices" else None

    def setProperty(self, name, value):
        self.log.append((name, value))

    def say(self, text):
        self.log.append(("say", text))

    def runAndWait(self):
        pass

    def stop(self):
        pass


class FakeSpeech:
    """Replaces the pyttsx3 module."""

    def __init__(self, voices):
        self.log = []
        self.voices = voices

    def init(self):
        return FakeEngine(self.log, self.voices)


def voice(id_, name):
    return SimpleNamespace(id=id_, name=name, languages=[])


EN = voice("TTS_MS_EN-US_DAVID_11.0", "Microsoft David Desktop - English (United States)")
HI = voice("TTS_MS_HI-IN_HEMANT_11.0", "Microsoft Hemant - Hindi (India)")
MR = voice("TTS_MS_MR-IN_AAROHI_11.0", "Microsoft Aarohi - Marathi (India)")


class ScriptedClient:
    """Fake Groq client. Each script item is a reply string or an exception to raise."""

    def __init__(self, *script):
        self.script = list(script)
        self.calls = []
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self.script.pop(0)
        if isinstance(item, Exception):
            raise item
        message = SimpleNamespace(content=item)
        return SimpleNamespace(choices=[SimpleNamespace(message=message, finish_reason="stop")])


def sdk_error(name):
    return type(name, (Exception,), {})("boom")


class Clock:
    def __init__(self):
        self.now = 1000.0

    def monotonic(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


class RunningCamera:
    """Plays the part of a running ASLCameraWorker for the window-level tests."""

    def __init__(self, window):
        self.window = window
        self.commands = []

    def isRunning(self):
        return True

    def enqueue_command(self, command, value=None):
        self.commands.append((command, value))
        text = self.window.recognized_text.toPlainText()
        if command == "space" and text and not text.endswith(" "):
            self.window.update_recognized_text(text + " ")
        elif command == "clear":
            self.window.update_recognized_text("")


def link(target):
    return SimpleNamespace(toString=lambda: target)


class WindowTests(unittest.TestCase):
    def setUp(self):
        import asl_studio
        from communication import GroqService, JobQueue
        qt_stub.QSettings.store = {}
        self.module = asl_studio
        self.clock = Clock()
        patcher = mock.patch.object(asl_studio, "time", self.clock)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.speech = FakeSpeech([EN, HI, MR])
        speech_patch = mock.patch.dict(sys.modules, {"pyttsx3": self.speech})
        speech_patch.start()
        self.addCleanup(speech_patch.stop)

        self.window = asl_studio.ASLStudio()
        self.window.queue = JobQueue(clock=self.clock.monotonic)
        self.GroqService = GroqService
        self.camera = RunningCamera(self.window)
        self.window.camera_worker = self.camera

    def use(self, *script):
        self.client = ScriptedClient(*script)
        self.window.service = self.GroqService(api_key="test", client=self.client)
        return self.client

    def sign_and_pause(self, text, seconds=6.0):
        """The recognizer commits `text`, the hand leaves, and time passes."""
        self.window.update_recognized_text(text)
        self.window._hand_present = False
        for _ in range(int(seconds / 0.1)):
            self.clock.advance(0.1)
            self.window._tick()

    def jobs(self):
        return [i["job"] for i in self.window.transcript_items if i["type"] == "job"]

    def notes(self):
        return [i["text"] for i in self.window.transcript_items if i["type"] == "note"]

    # ------------------------------------------------------------------ workflow, English

    def test_english_message_is_sent_automatically_and_answered_and_spoken(self):
        client = self.use("Mn.", "Hello! How can I help?")
        self.sign_and_pause("MN")

        job, = self.jobs()
        self.assertEqual((job.state, job.corrected, job.reply), ("done", "Mn.", "Hello! How can I help?"))
        self.assertEqual(len(client.calls), 2)                    # one correction, one reply
        self.assertEqual(client.calls[0]["messages"][1]["content"], "<signed>MN</signed>")
        # The assistant gets the corrected sentence as an ordinary user message.
        self.assertEqual(client.calls[1]["messages"][-1], {"role": "user", "content": "Mn."})
        self.assertEqual(self.window.recognized_text.toPlainText(), "")
        self.assertIn(("space", None), self.camera.commands)
        self.assertEqual(self.camera.commands.count(("clear", None)), 1)
        self.assertEqual(self.speech.log, [("voice", EN.id), ("say", "Hello! How can I help?")])
        self.assertEqual(len(self.window.history), 2)

    def test_letters_are_never_sent_individually(self):
        client = self.use("Hi.", "Hello.")
        for text in ("H", "HI"):
            self.window.update_recognized_text(text)
            self.window._hand_present = True
            for _ in range(10):
                self.clock.advance(0.1)
                self.window._tick()
        self.assertEqual(client.calls, [])                        # still spelling
        self.sign_and_pause("HI")
        self.assertEqual(len(client.calls), 2)

    def test_no_duplicate_submission(self):
        client = self.use("Hi.", "Hello.")
        self.sign_and_pause("HI", seconds=4.0)
        self.assertEqual(len(self.jobs()), 1)
        for _ in range(100):                                      # long idle afterwards
            self.clock.advance(0.1)
            self.window._tick()
        self.window._send_signed(auto=False)                      # Send now on empty text
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(len(client.calls), 2)

    def test_double_press_of_send_now_submits_once(self):
        client = self.use("Hi.", "Hello.")
        self.window.update_recognized_text("HI")
        self.window._send_signed(auto=False)
        self.window.update_recognized_text("HI")                  # same letters again at once
        self.window._send_signed(auto=False)
        self.assertEqual(len(self.jobs()), 1)
        self.assertEqual(len(client.calls), 2)

    def test_empty_and_punctuation_only_messages_are_not_sent(self):
        client = self.use("x")
        self.sign_and_pause("")
        self.sign_and_pause(". ,")
        self.window.update_recognized_text(".")
        self.window._send_signed(auto=False)
        self.assertEqual((self.jobs(), client.calls), ([], []))

    def test_typing_in_the_box_holds_off_auto_send(self):
        client = self.use("Hi.", "Hello.")
        self.window.update_recognized_text("HI")
        self.window.recognized_text._focus = True
        self.window._last_edit = self.clock.now
        for _ in range(40):                                       # 4 s, longer than the delay
            self.clock.advance(0.1)
            self.window._tick()
        self.assertEqual(client.calls, [])

    # --------------------------------------------------------------------- languages

    def test_hindi_flow_translates_replies_and_speaks_with_hindi_voice(self):
        client = self.use("नमस्ते, आप कैसे हैं?", "मैं ठीक हूँ। आप कैसे हैं?")
        self.window._set_language("hi")
        self.sign_and_pause("HELLO HOW ARE YOU")

        self.assertIn("Hindi", client.calls[0]["messages"][0]["content"])
        self.assertIn("Devanagari", client.calls[0]["messages"][0]["content"])
        self.assertIn("Always reply in Hindi", client.calls[1]["messages"][0]["content"])
        self.assertEqual(self.jobs()[0].corrected, "नमस्ते, आप कैसे हैं?")
        self.assertEqual(self.speech.log, [("voice", HI.id), ("say", "मैं ठीक हूँ। आप कैसे हैं?")])
        self.assertEqual(qt_stub.QSettings.store["language"], "hi")

    def test_marathi_flow_uses_marathi_voice(self):
        client = self.use("नमस्कार, तुम्ही कसे आहात?", "मी ठीक आहे.")
        self.window._set_language("mr")
        self.sign_and_pause("HELLO HOW ARE YOU")
        self.assertIn("Marathi", client.calls[0]["messages"][0]["content"])
        self.assertIn("Always reply in Marathi", client.calls[1]["messages"][0]["content"])
        self.assertEqual(self.speech.log[0], ("voice", MR.id))

    def test_marathi_falls_back_to_hindi_voice_and_tells_the_user(self):
        self.speech.voices = [EN, HI]
        self.use("नमस्कार.", "मी ठीक आहे.")
        self.window._set_language("mr")
        self.sign_and_pause("HELLO")
        self.assertEqual(self.speech.log[0], ("voice", HI.id))
        self.assertTrue(any("Hindi voice" in n for n in self.notes()))

    def test_missing_voice_is_reported_and_nothing_is_spoken_in_the_wrong_language(self):
        self.speech.voices = [EN]
        self.use("नमस्ते.", "ठीक है।")
        self.window._set_language("hi")
        self.sign_and_pause("HELLO")
        self.assertEqual([e for e in self.speech.log if e[0] == "say"], [])
        self.assertTrue(any("No Hindi voice" in n for n in self.notes()))
        self.assertEqual(self.jobs()[0].state, "done")             # the reply still shows

    def test_language_chosen_at_send_time_is_kept_for_that_message(self):
        client = self.use("Hi.", "Hello.")
        self.window.update_recognized_text("HI")
        self.window._send_signed(auto=False)
        self.window._set_language("hi")                            # changed afterwards
        self.assertEqual(self.jobs()[0].language, "en")
        self.assertIn("English", client.calls[1]["messages"][0]["content"])

    def test_speak_replies_switch(self):
        self.use("Hi.", "Hello.")
        self.window.speak_checkbox.setChecked(False)
        self.sign_and_pause("HI")
        self.assertEqual(self.speech.log, [])

    def test_speak_link_uses_the_messages_own_language(self):
        self.use("नमस्ते.", "ठीक है।")
        self.window._set_language("hi")
        self.sign_and_pause("HELLO")
        self.speech.log.clear()
        self.window._set_language("en")
        self.window._transcript_link(link(f"speak:{self.jobs()[0].id}:user"))
        self.assertEqual(self.speech.log, [("voice", HI.id), ("say", "नमस्ते.")])

    # ----------------------------------------------------------------------- failures

    def test_failure_is_shown_and_retry_succeeds(self):
        self.use(sdk_error("APIConnectionError"))
        self.sign_and_pause("HELLO")
        job, = self.jobs()
        self.assertEqual(job.state, "failed")
        self.assertIn("Could not reach", job.error)
        self.assertIn("retry:", self.module.render_transcript(self.window.transcript_items))
        self.assertEqual(self.window.recognized_text.toPlainText(), "")   # raw text is on screen

        client = self.use("Hello.", "Hi!")
        self.window._transcript_link(link(f"retry:{job.id}"))
        self.assertEqual((job.state, job.reply), ("done", "Hi!"))
        self.assertEqual(len(client.calls), 2)

    def test_retry_after_reply_failure_does_not_repeat_correction(self):
        self.use("Hello.", sdk_error("RateLimitError"))
        self.sign_and_pause("HELLO")
        job, = self.jobs()
        self.assertEqual((job.state, job.corrected), ("failed", "Hello."))
        self.assertIn("rate-limiting", job.error)
        self.assertEqual(self.window.history, [])                  # failed turns are not remembered

        client = self.use("Hi!")
        self.window._transcript_link(link(f"retry:{job.id}"))
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(job.state, "done")

    def test_unreadable_signing_never_reaches_the_assistant(self):
        client = self.use("UNCLEAR")
        self.sign_and_pause("QXZ")
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(self.jobs()[0].state, "unclear")
        self.assertEqual(self.speech.log, [])

    def test_missing_api_key_keeps_the_text_and_does_not_loop(self):
        self.window.service = self.GroqService(api_key="")
        self.window.update_recognized_text("HELLO")
        for _ in range(80):
            self.clock.advance(0.1)
            self.window._tick()
        self.assertEqual(self.jobs(), [])
        self.assertEqual(self.window.recognized_text.toPlainText(), "HELLO")
        self.window._send_signed(auto=False)
        self.window._send_signed(auto=False)
        self.assertEqual(len([n for n in self.notes() if "GROQ_API_KEY" in n]), 1)
        self.assertEqual(self.window.recognized_text.toPlainText(), "HELLO")

    def test_speech_engine_failure_is_reported_not_raised(self):
        self.use("Hi.", "Hello.")
        self.speech.init = mock.Mock(side_effect=RuntimeError("no audio device"))
        self.sign_and_pause("HI")
        self.assertTrue(any("no audio device" in n for n in self.notes()))
        self.assertEqual(self.jobs()[0].state, "done")

    # ------------------------------------------------------------------ typed + housekeeping

    def test_typed_message_uses_the_selected_language_without_correction(self):
        client = self.use("ठीक है।")
        self.window._set_language("hi")
        self.window.chat_input.setText("What is the weather")
        self.window._send_typed()
        self.assertEqual(len(client.calls), 1)
        self.assertEqual(client.calls[0]["messages"][-1]["content"], "What is the weather")
        self.assertIn("Always reply in Hindi", client.calls[0]["messages"][0]["content"])
        self.assertEqual(self.window.chat_input.text(), "")

    def test_empty_typed_message_is_ignored(self):
        client = self.use("x")
        self.window.chat_input.setText("   ")
        self.window._send_typed()
        self.assertEqual((self.jobs(), client.calls), ([], []))

    def test_conversation_context_carries_across_messages(self):
        client = self.use("My name is Sam.", "Nice to meet you, Sam.", "Do I know you?", "You are Sam.")
        self.sign_and_pause("MY NAME IS SAM")
        self.clock.advance(5)
        self.sign_and_pause("DO I KNOW YOU")
        roles = [m["role"] for m in client.calls[3]["messages"]]
        self.assertEqual(roles, ["system", "user", "assistant", "user"])

    def test_clear_conversation_resets_history_and_ignores_late_results(self):
        self.use("Hi.", "Hello.")
        self.sign_and_pause("HI")
        self.window.clear_conversation()
        self.assertEqual((self.window.history, self.window.transcript_items), ([], []))
        self.window._job_succeeded(999, "late reply")              # a result for a cleared job
        self.assertEqual(self.window.history, [])

    def test_markdown_is_not_read_aloud(self):
        self.use("Hi.", "**Sure!** Here is a *list*:\n- one\n- two")
        self.sign_and_pause("HI")
        spoken = [e[1] for e in self.speech.log if e[0] == "say"][0]
        self.assertNotIn("*", spoken)
        self.assertNotIn("- ", spoken)

    # --------------------------------------------------------------------- interface

    def test_removed_labels_do_not_appear_anywhere_in_the_interface_code(self):
        source = (ROOT / "asl_studio.py").read_text(encoding="utf-8").lower()
        for label in ("intelligent sign communication studio", "asl genai studio",
                      "mediapipe · onnx runtime", "system ready", "pyttsx3 ·", "● local mode"):
            self.assertNotIn(label, source)

    def test_no_label_text_leaks_into_rendered_transcript_or_widgets(self):
        self.use("Hi.", "Hello.")
        self.sign_and_pause("HI")
        rendered = self.module.render_transcript(self.window.transcript_items).lower()
        self.assertNotIn("system ready", rendered)
        self.assertNotIn("genai", rendered)

    def test_stylesheet_is_complete(self):
        sheet = self.module.build_stylesheet()
        self.assertNotIn("@", sheet)
        self.assertEqual(sheet.count("{"), sheet.count("}"))

    def test_empty_transcript_explains_what_to_do(self):
        self.assertIn("Fingerspell", self.module.render_transcript([]))

    def test_transcript_escapes_html(self):
        from communication import Job
        job = Job(1, "<b>X</b>", "en", "typed", corrected="<b>X</b>", reply="<i>y</i>", state="done")
        page = self.module.render_transcript([{"type": "job", "job": job}])
        self.assertNotIn("<b>X</b>", page)
        self.assertIn("&lt;b&gt;X&lt;/b&gt;", page)


# ------------------------------------------------------------------ camera + recognition

try:
    import cv2  # noqa: F401
    import numpy as np
    import onnxruntime  # noqa: F401
    HAVE_VISION = True
except ImportError:
    HAVE_VISION = False


class FakeCapture:
    def __init__(self, opened=True, frames=True):
        self.opened, self.frames, self.released = opened, frames, False

    def isOpened(self):
        return self.opened

    def read(self):
        if not self.frames:
            return False, None
        return True, np.zeros((480, 640, 3), np.uint8)

    def release(self):
        self.released = True


class ScriptedHands:
    """Stands in for mp.solutions.hands.Hands: plays a script of poses, then stops."""

    def __init__(self, script, poses):
        self.script = list(script)
        self.poses = poses
        self.worker = None
        self.frame = 0

    def __call__(self, **kwargs):
        return self

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def process(self, rgb):
        if self.frame >= len(self.script):
            self.worker.stop()
            return SimpleNamespace(multi_hand_landmarks=None)
        pose = self.script[self.frame]
        self.frame += 1
        if pose is None:
            return SimpleNamespace(multi_hand_landmarks=None)
        pts = np.array(self.poses[pose][1])
        marks = [SimpleNamespace(x=(px * 100 + 320) / 640, y=(py * 100 + 300) / 480) for px, py in pts]
        return SimpleNamespace(multi_hand_landmarks=[SimpleNamespace(landmark=marks)])


def fake_mediapipe(hands):
    return SimpleNamespace(solutions=SimpleNamespace(
        hands=SimpleNamespace(Hands=hands, HAND_CONNECTIONS=None),
        drawing_utils=SimpleNamespace(draw_landmarks=lambda *a, **k: None)))


@unittest.skipUnless(HAVE_VISION, "needs opencv-python, numpy and onnxruntime")
class RecognitionTests(unittest.TestCase):
    """Runs the real camera loop and the real ONNX model on scripted hand poses."""

    @classmethod
    def setUpClass(cls):
        cls.poses = json.loads((TESTS / "fixtures" / "synthetic_poses.json").read_text())

    def setUp(self):
        import asl_studio
        self.module = asl_studio

    def run_worker(self, script=None, capture=None, patches=()):
        hands = ScriptedHands(script or [], self.poses)
        capture = capture or FakeCapture()
        worker = self.module.ASLCameraWorker()
        hands.worker = worker
        seen = {"text": [], "error": [], "hand": [], "pred": [], "frames": 0}
        worker.text_updated.connect(seen["text"].append)
        worker.camera_error.connect(seen["error"].append)
        worker.hand_state.connect(seen["hand"].append)
        worker.prediction.connect(lambda *a: seen["pred"].append(a))
        worker.frame_ready.connect(lambda _image: seen.__setitem__("frames", seen["frames"] + 1))
        with mock.patch.object(self.module, "mp", fake_mediapipe(hands)), \
                mock.patch.object(self.module.cv2, "VideoCapture", return_value=capture):
            for patch in patches:
                patch.start()
            try:
                worker.run()
            finally:
                for patch in patches:
                    patch.stop()
        return worker, seen, capture

    def test_onnx_model_loads_with_expected_shape(self):
        from realtime_translator import load_session
        session, input_name, size, outputs, taps = load_session(str(self.module.MODEL_PATH))
        self.assertEqual((input_name, size, "logits" in outputs), ("input_image", 96, True))
        classes = self.module.CLASS_NAMES_PATH.read_text().strip().split(",")
        self.assertEqual(len(classes), 26)
        zeros = np.zeros((1, 96, 96, 3), np.float32)
        self.assertEqual(session.run(["logits"], {input_name: zeros})[0].shape, (1, 26))

    def test_fingerspelling_commits_letters_through_the_real_model(self):
        script = ["m"] * 25 + [None] * 4 + ["n"] * 25 + [None] * 4
        worker, seen, capture = self.run_worker(script)
        self.assertEqual(seen["error"], [])
        self.assertEqual(worker.buffer.text, "MN")
        self.assertEqual(seen["text"][-1], "MN")
        self.assertGreater(seen["frames"], 50)
        self.assertIn(True, seen["hand"])
        self.assertEqual(seen["hand"][-1], False)
        self.assertTrue(any(p[0] == "M" and p[1] >= 0.75 for p in seen["pred"]))
        self.assertTrue(capture.released)

    def test_text_is_handed_to_the_worker_and_commands_apply(self):
        worker = self.module.ASLCameraWorker()
        worker.buffer.text = "HI"
        worker.enqueue_command("space")
        worker.enqueue_command("backspace")
        worker.enqueue_command("repeat")
        worker.enqueue_command("punctuation", ".")
        hands = ScriptedHands([], self.poses)
        hands.worker = worker
        with mock.patch.object(self.module, "mp", fake_mediapipe(hands)), \
                mock.patch.object(self.module.cv2, "VideoCapture", return_value=FakeCapture()):
            worker.run()
        # "HI" -> space -> backspace -> repeat last letter -> full stop
        self.assertEqual(worker.buffer.text, "HII.")

    def test_webcam_unavailable_gives_a_clear_error(self):
        worker, seen, capture = self.run_worker(capture=FakeCapture(opened=False))
        self.assertEqual(len(seen["error"]), 1)
        self.assertIn("Could not access the webcam", seen["error"][0])
        self.assertTrue(capture.released)

    def test_camera_that_stops_delivering_frames_is_reported(self):
        stall = mock.patch.object(self.module, "CAMERA_STALL_SECONDS", 0.05)
        worker, seen, capture = self.run_worker(capture=FakeCapture(frames=False), patches=[stall])
        self.assertEqual(len(seen["error"]), 1)
        self.assertIn("stopped sending images", seen["error"][0])
        self.assertTrue(capture.released)

    def test_missing_model_gives_a_clear_error(self):
        missing = mock.patch.object(self.module, "MODEL_PATH", ROOT / "does_not_exist.onnx")
        worker, seen, _ = self.run_worker(patches=[missing])
        self.assertIn("ONNX model not found", seen["error"][0])

    def test_mediapipe_without_solutions_gives_an_actionable_error(self):
        worker = self.module.ASLCameraWorker()
        errors = []
        worker.camera_error.connect(errors.append)
        with mock.patch.object(self.module, "mp", SimpleNamespace()):
            worker.run()
        self.assertIn("pip install mediapipe", errors[0])


if __name__ == "__main__":
    unittest.main()
