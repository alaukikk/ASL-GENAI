"""Communication Studio.

Desktop application that combines the repository's MediaPipe + ONNX ASL fingerspelling
recognizer with Groq text assistance and offline speech.

Fingerspell a message: when you pause, it is corrected (and translated for Hindi or
Marathi) by Groq, sent to the assistant as a normal message, and the reply is spoken in
the selected language. The workflow logic lives in communication.py; this file is the
camera pipeline and the interface.

Run from this directory with: python asl_studio.py
"""

from __future__ import annotations

import html
import math
import sys
import time
from pathlib import Path
from queue import SimpleQueue

import cv2
import mediapipe as mp
import numpy as np

from PySide6.QtCore import QSettings, Qt, QThread, QTimer, Signal
from PySide6.QtGui import QImage, QPixmap, QTextCursor
from PySide6.QtWidgets import (
    QApplication,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QTextBrowser,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

import preprocessing as pp
from communication import (
    DEFAULT_LANGUAGE,
    HISTORY_LIMIT,
    LANGUAGES,
    AIError,
    GroqService,
    JobQueue,
    MessageSegmenter,
    UnclearMessage,
    friendly_error,
    get_language,
    run_job,
    select_voice,
    speakable,
    trim_history,
)
from realtime_translator import (
    CONV_TAP,
    DENSE_TAP,
    LOGITS,
    SMOOTHING_WINDOW,
    SpellingBuffer,
    load_session,
)
from motion import MotionTracker
from smoothing import OneEuroFilter


BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "asl_cnn_model.onnx"
CLASS_NAMES_PATH = BASE_DIR / "class_names.txt"

# Pauses that turn fingerspelled letters into words and finished messages.
WORD_GAP_SECONDS = 1.2
SEND_DELAY_CHOICES = (2, 3, 5, 8)      # seconds of silence before a message is sent
DEFAULT_SEND_DELAY = 3
EDIT_HOLD_SECONDS = 8.0                # no auto-send this long after typing in the box
CAMERA_STALL_SECONDS = 3.0             # camera open but no frames for this long = error


# ------------------------------------------------------------------------------ recognition

class ASLCameraWorker(QThread):
    """Owns the webcam and recognition pipeline outside the GUI thread."""

    frame_ready = Signal(QImage)
    text_updated = Signal(str)
    status_updated = Signal(str)
    camera_error = Signal(str)
    hand_state = Signal(bool)
    prediction = Signal(str, float, float)      # letter, confidence, hold progress 0..1

    def __init__(self, camera_index: int = 0):
        super().__init__()
        self._running = True
        self.camera_index = camera_index
        self.buffer = SpellingBuffer()
        self._commands = SimpleQueue()

    def enqueue_command(self, command: str, value: str | None = None):
        self._commands.put((command, value))

    def stop(self):
        self._running = False

    def run(self):
        capture = None
        last_emitted_text = None
        try:
            self.status_updated.emit("Starting camera…")

            if not CLASS_NAMES_PATH.is_file():
                raise FileNotFoundError(f"Class list not found: {CLASS_NAMES_PATH}")
            if not MODEL_PATH.is_file():
                raise FileNotFoundError(f"ONNX model not found: {MODEL_PATH}")
            if not hasattr(mp, "solutions"):
                raise RuntimeError(
                    "The installed MediaPipe has no hand-tracking 'solutions' API. "
                    "Install a compatible release, for example: pip install mediapipe==0.10.21"
                )

            class_names = CLASS_NAMES_PATH.read_text(encoding="utf-8").strip().split(",")
            session, input_name, size, output_names, has_taps = load_session(str(MODEL_PATH))
            wanted = [LOGITS if LOGITS in output_names else output_names[0]]
            if has_taps:
                wanted.extend([CONV_TAP, DENSE_TAP])

            tracker = MotionTracker()
            smoother = OneEuroFilter()
            score_history = []

            capture = cv2.VideoCapture(self.camera_index)
            if not capture.isOpened():
                raise RuntimeError(
                    "Could not access the webcam. Check camera permissions or close other "
                    "apps that are using it."
                )

            self.status_updated.emit("Camera running")
            self.text_updated.emit(self.buffer.text)
            last_emitted_text = self.buffer.text
            last_hand = None
            last_prediction = None
            last_frame_time = time.monotonic()

            with mp.solutions.hands.Hands(
                model_complexity=pp.HAND_MODEL_COMPLEXITY,
                max_num_hands=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5,
            ) as hands:
                while self._running:
                    while not self._commands.empty():
                        command, value = self._commands.get()
                        if command == "set_text":
                            self.buffer.text = value or ""
                            self.buffer._last_committed = None
                        elif command == "clear":
                            self.buffer.clear()
                        elif command == "space":
                            self.buffer.space()
                        elif command == "backspace":
                            self.buffer.backspace()
                        elif command == "repeat":
                            self.buffer.repeat()
                        elif command == "punctuation":
                            self.buffer.punctuation(value or ".")

                    ok, frame = capture.read()
                    if not ok:
                        if time.monotonic() - last_frame_time > CAMERA_STALL_SECONDS:
                            raise RuntimeError(
                                "The camera stopped sending images. It may have been "
                                "disconnected or taken by another app."
                            )
                        self.msleep(10)
                        continue
                    last_frame_time = time.monotonic()

                    frame = cv2.flip(frame, 1)
                    frame.flags.writeable = False
                    results = hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    frame.flags.writeable = True
                    height, width = frame.shape[:2]

                    if results.multi_hand_landmarks:
                        landmarks = results.multi_hand_landmarks[0]
                        points_px = smoother(pp.landmarks_to_pixels(landmarks, width, height))
                        tracker.update(points_px)
                        normalized = pp.normalize_landmarks(points_px)
                        skeleton_rgb = pp.render_skeleton(normalized, size=size)
                        outputs = session.run(wanted, {input_name: pp.model_input(skeleton_rgb)})

                        logits = outputs[0][0]
                        exponentials = np.exp(logits - logits.max())
                        score_history.append(exponentials / exponentials.sum())
                        if len(score_history) > SMOOTHING_WINDOW:
                            score_history.pop(0)
                        scores = np.mean(score_history, axis=0)

                        letter = class_names[int(np.argmax(scores))]
                        confidence = float(scores.max())
                        resolved = tracker.classify(letter)
                        if resolved and resolved in class_names:
                            letter = resolved

                        moving = tracker.is_moving()
                        self.buffer.update(letter, confidence, True, moving)

                        mp.solutions.drawing_utils.draw_landmarks(
                            frame, landmarks, mp.solutions.hands.HAND_CONNECTIONS
                        )
                        current = (letter.upper(), round(confidence, 2), round(self.buffer.progress(), 1))
                    else:
                        score_history.clear()
                        smoother.reset()
                        tracker.reset()
                        self.buffer.update(None, 0.0, False, False)
                        current = ("", 0.0, 0.0)

                    hand_present = bool(results.multi_hand_landmarks)
                    if hand_present != last_hand:
                        self.hand_state.emit(hand_present)
                        last_hand = hand_present
                    if current != last_prediction:
                        self.prediction.emit(*current)
                        last_prediction = current

                    # Emit only when a letter is committed or the buffer otherwise changes.
                    # This avoids resetting the text cursor on every camera frame.
                    if self.buffer.text != last_emitted_text:
                        self.text_updated.emit(self.buffer.text)
                        last_emitted_text = self.buffer.text

                    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    h, w, channels = rgb_frame.shape
                    image = QImage(
                        rgb_frame.data,
                        w,
                        h,
                        channels * w,
                        QImage.Format.Format_RGB888,
                    ).copy()
                    self.frame_ready.emit(image)

        except Exception as error:
            self.camera_error.emit(str(error) or type(error).__name__)
        finally:
            if capture is not None:
                capture.release()
            self.hand_state.emit(False)


# ----------------------------------------------------------------------------- AI + speech

class PipelineWorker(QThread):
    """Runs one message (correction, then the assistant's reply) off the UI thread."""

    corrected = Signal(int, str)
    succeeded = Signal(int, str)
    unclear = Signal(int)
    failed = Signal(int, str)

    def __init__(self, service: GroqService, job, history: list[dict], parent=None):
        super().__init__(parent)
        self.service = service
        self.job = job
        self.history = history

    def run(self):
        job = self.job
        try:
            reply = run_job(
                self.service, job, self.history,
                on_corrected=lambda j: self.corrected.emit(j.id, j.corrected),
            )
            self.succeeded.emit(job.id, reply)
        except UnclearMessage:
            self.unclear.emit(job.id)
        except AIError as error:
            self.failed.emit(job.id, str(error))
        except Exception as error:
            self.failed.emit(job.id, friendly_error(error))


class SpeechWorker(QThread):
    """Speaks one message with an installed offline voice that matches its language."""

    speech_failed = Signal(str)
    voice_fallback = Signal(str)

    def __init__(self, text: str, language_code: str, parent=None):
        super().__init__(parent)
        self.text = text
        self.language = get_language(language_code)
        self._engine = None

    def stop_speech(self):
        # pyttsx3 exposes stop() for interrupting the current utterance.
        engine = self._engine
        if engine is not None:
            try:
                engine.stop()
            except Exception:
                pass

    def run(self):
        try:
            try:
                import pyttsx3
            except ImportError:
                self.speech_failed.emit("Speech needs the 'pyttsx3' package. Run: pip install pyttsx3")
                return
            self._engine = pyttsx3.init()

            try:
                voices = self._engine.getProperty("voices") or []
            except Exception:
                voices = []
            voice, matched = select_voice(voices, self.language.code)
            if voice is None and self.language.code != "en":
                self.speech_failed.emit(
                    f"No {self.language.name} voice is installed, so this was not spoken. "
                    "Add one in your system's speech settings "
                    "(Windows: Settings › Time & language › Speech)."
                )
                return
            if voice is not None:
                self._engine.setProperty("voice", voice.id)
                if matched != self.language.code:
                    self.voice_fallback.emit(
                        f"No {self.language.name} voice is installed. "
                        f"Using a {get_language(matched).name} voice instead."
                    )

            self._engine.say(speakable(self.text))
            self._engine.runAndWait()
        except Exception as error:
            self.speech_failed.emit(f"Speech error: {error}")
        finally:
            if self._engine is not None:
                try:
                    self._engine.stop()
                except Exception:
                    pass
                self._engine = None


# ------------------------------------------------------------------------------------ look

PALETTE = {
    "BG": "#F2F3F5",
    "CARD": "#FFFFFF",
    "BORDER": "#DCDFE4",
    "BORDER_STRONG": "#C3C8D0",
    "TEXT": "#1B1F24",
    "MUTED": "#5C6572",
    "DISABLED": "#9AA1AC",
    "ACCENT": "#24407A",
    "ACCENT_HOVER": "#1B3262",
    "ACCENT_SOFT": "#D9E2F4",
    "HOVER": "#F0F2F5",
    "PRESSED": "#E4E7EC",
    "TRACK": "#E3E6EB",
    "CAMERA": "#15181D",
    "ERROR": "#A8281F",
    "WARN_BG": "#FFF6E5",
    "WARN_BORDER": "#EBCB8B",
    "WARN_TEXT": "#6B4300",
    "BAR_NEUTRAL": "#C3C8D0",
}

STYLESHEET = """
* { font-family: "Segoe UI"; font-size: 13px; color: @TEXT@; }
QMainWindow, QWidget#Root { background: @BG@; }
QLabel { background: transparent; }

QFrame#Toolbar { background: @CARD@; border: none; border-bottom: 1px solid @BORDER@; }
QFrame#Card { background: @CARD@; border: 1px solid @BORDER@; border-radius: 6px; }

QLabel#CardTitle { font-size: 14px; font-weight: 600; }
QLabel#Field { color: @MUTED@; font-size: 12px; }
QLabel#Small { color: @MUTED@; font-size: 12px; }
QLabel#CameraPreview { background: @CAMERA@; color: #AEB6C2; font-size: 14px; border: none; }
QLabel#Notice { background: @WARN_BG@; color: @WARN_TEXT@; border: 1px solid @WARN_BORDER@;
    border-radius: 4px; padding: 8px 10px; }

QPushButton { background: @CARD@; border: 1px solid @BORDER_STRONG@; border-radius: 4px;
    padding: 7px 14px; }
QPushButton:hover { background: @HOVER@; }
QPushButton:pressed { background: @PRESSED@; }
QPushButton:focus { border-color: @ACCENT@; }
QPushButton:disabled { color: @DISABLED@; background: @BG@; border-color: @BORDER@; }
QPushButton#Primary { background: @ACCENT@; border-color: @ACCENT@; color: #FFFFFF;
    font-weight: 600; }
QPushButton#Primary:hover { background: @ACCENT_HOVER@; border-color: @ACCENT_HOVER@; }
QPushButton#Primary:disabled { background: #AEB7C7; border-color: #AEB7C7; color: #FFFFFF; }
QPushButton#Quiet { border: none; background: transparent; color: @MUTED@; padding: 4px 8px; }
QPushButton#Quiet:hover { color: @TEXT@; background: @HOVER@; }

QPushButton#SegFirst, QPushButton#SegMid, QPushButton#SegLast { padding: 6px 16px; }
QPushButton#SegFirst { border-top-right-radius: 0; border-bottom-right-radius: 0; }
QPushButton#SegMid { border-radius: 0; border-left: none; }
QPushButton#SegLast { border-top-left-radius: 0; border-bottom-left-radius: 0;
    border-left: none; }
QPushButton#SegFirst:checked, QPushButton#SegMid:checked, QPushButton#SegLast:checked {
    background: @ACCENT@; border-color: @ACCENT@; color: #FFFFFF; font-weight: 600; }

QTextEdit, QLineEdit { background: @CARD@; border: 1px solid @BORDER_STRONG@;
    border-radius: 4px; padding: 8px 10px; selection-background-color: @ACCENT_SOFT@;
    selection-color: @TEXT@; }
QTextEdit:focus, QLineEdit:focus { border-color: @ACCENT@; }
QTextEdit#Composer { font-size: 20px; }
QTextBrowser#Transcript { background: @CARD@; border: none; padding: 0; }

QComboBox { background: @CARD@; border: 1px solid @BORDER_STRONG@; border-radius: 4px;
    padding: 5px 10px; }
QComboBox:hover { background: @HOVER@; }
QCheckBox { spacing: 8px; }

QProgressBar { border: none; background: @TRACK@; min-height: 4px; max-height: 4px;
    border-radius: 2px; }
QProgressBar::chunk { background: @ACCENT@; border-radius: 2px; }

QScrollBar:vertical { background: transparent; width: 10px; margin: 0; }
QScrollBar::handle:vertical { background: #C9CED6; border-radius: 4px; min-height: 28px; }
QScrollBar::handle:vertical:hover { background: #AEB5C0; }
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0; }
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: transparent; }
"""


def build_stylesheet() -> str:
    sheet = STYLESHEET
    for key, value in PALETTE.items():
        sheet = sheet.replace(f"@{key}@", value)
    return sheet


def _esc(text: str) -> str:
    return html.escape(text).replace("\n", "<br>")


def render_transcript(items: list[dict]) -> str:
    """The conversation as rich text for the transcript view."""
    c = PALETTE

    def entry(label: str, bar: str, body: str) -> str:
        return (
            '<table width="100%" cellspacing="0" cellpadding="0"><tr>'
            f'<td width="3" bgcolor="{bar}"></td>'
            '<td width="12"></td>'
            f'<td><p style="margin-top:0; margin-bottom:0; font-size:11px; font-weight:bold; color:{c["MUTED"]}">{label}</p>'
            f"{body}</td></tr></table>"
            '<p style="margin-top:0; margin-bottom:0; font-size:8px">&nbsp;</p>'
        )

    def text_block(text: str, color: str | None = None, size: int = 15, italic: bool = False) -> str:
        style = f"margin-top:2px; margin-bottom:0; font-size:{size}px;"
        if color:
            style += f" color:{color};"
        if italic:
            style += " font-style:italic;"
        return f'<p style="{style}">{_esc(text)}</p>'

    def link(href: str, label: str) -> str:
        return f'<p style="margin-top:4px; margin-bottom:0; font-size:12px"><a href="{href}">{label}</a></p>'

    if not items:
        return (
            f'<p style="color:{c["MUTED"]}; font-size:13px; margin-top:6px">'
            "Fingerspell a message and pause. It is corrected, sent to the assistant and "
            "answered here automatically. You can also type a message below.</p>"
        )

    parts: list[str] = []
    for item in items:
        if item["type"] == "note":
            color = c["ERROR"] if item.get("level") == "error" else c["MUTED"]
            parts.append(
                f'<p style="margin-top:0; margin-bottom:0; font-size:12px; color:{color}">{_esc(item["text"])}</p>'
                '<p style="margin-top:0; margin-bottom:0; font-size:8px">&nbsp;</p>'
            )
            continue

        job = item["job"]
        # The signer's message.
        shown = job.corrected
        if shown is None and job.state in ("failed", "unclear"):
            shown = job.raw
        if shown is None:
            body = text_block("Interpreting your signing…", c["MUTED"], 14, italic=True)
        else:
            body = text_block(shown)
            if job.kind == "signed" and job.corrected is not None:
                body += text_block(f"Signed: {job.raw}", c["MUTED"], 12)
            body += link(f"speak:{job.id}:user", "Speak")
        parts.append(entry("You", c["ACCENT"], body))

        # The assistant's answer.
        if job.state == "done" and job.reply:
            body = text_block(job.reply) + link(f"speak:{job.id}:reply", "Speak")
            parts.append(entry("Assistant", c["BAR_NEUTRAL"], body))
        elif job.state == "replying":
            parts.append(entry("Assistant", c["BAR_NEUTRAL"], text_block("Thinking…", c["MUTED"], 14, italic=True)))
        elif job.state == "unclear":
            parts.append(entry(
                "Assistant", c["BAR_NEUTRAL"],
                text_block("That signing could not be read as a message. Please try again.", c["MUTED"], 14),
            ))
        elif job.state == "failed":
            body = text_block(job.error or "Something went wrong.", c["ERROR"], 14)
            body += link(f"retry:{job.id}", "Try again")
            parts.append(entry("Assistant", c["ERROR"], body))
    return "".join(parts)


# ------------------------------------------------------------------------------- the window

class ASLStudio(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Communication Studio")
        self.resize(1320, 820)
        self.setMinimumSize(1040, 700)

        self.settings = QSettings("CommunicationStudio", "Studio")
        stored_language = str(self.settings.value("language", DEFAULT_LANGUAGE))
        self.language = stored_language if stored_language in LANGUAGES else DEFAULT_LANGUAGE

        self.service = GroqService()
        self.queue = JobQueue()
        self.segmenter = MessageSegmenter(WORD_GAP_SECONDS, float(DEFAULT_SEND_DELAY))
        self.history: list[dict] = []
        self.transcript_items: list[dict] = []
        self.camera_worker = None
        self.pipeline_worker = None
        self.speech_worker = None
        self._speech_next = None
        self._shown_notes: set[str] = set()
        self._hand_present = False
        self._camera_error = ""
        self._setting_recognized_text = False
        self._last_edit = float("-inf")     # when the person last typed in the signed text

        self.setStyleSheet(build_stylesheet())
        self._create_interface()
        self._restore_settings()
        self._refresh_transcript()

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(100)
        self.tick_timer.timeout.connect(self._tick)
        self.tick_timer.start()

    # ---- construction ------------------------------------------------------------------

    @staticmethod
    def _card() -> QFrame:
        card = QFrame()
        card.setObjectName("Card")
        return card

    def _create_interface(self):
        root = QWidget()
        root.setObjectName("Root")
        self.setCentralWidget(root)
        outer = QVBoxLayout(root)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        outer.addWidget(self._create_toolbar())

        body = QHBoxLayout()
        body.setContentsMargins(20, 20, 20, 20)
        body.setSpacing(20)
        left = QVBoxLayout()
        left.setSpacing(16)
        left.addWidget(self._create_camera_card(), 1)
        left.addWidget(self._create_composer_card())
        body.addLayout(left, 6)
        body.addWidget(self._create_conversation_card(), 5)
        outer.addLayout(body, 1)

        self._connect_signals()

    def _create_toolbar(self) -> QFrame:
        toolbar = QFrame()
        toolbar.setObjectName("Toolbar")
        row = QHBoxLayout(toolbar)
        row.setContentsMargins(20, 12, 20, 12)
        row.setSpacing(12)

        label = QLabel("Language")
        label.setObjectName("Field")
        row.addWidget(label)

        segment = QHBoxLayout()
        segment.setSpacing(0)
        self.language_group = QButtonGroup(self)
        self.language_group.setExclusive(True)
        self.language_buttons: dict[str, QPushButton] = {}
        last_index = len(LANGUAGES) - 1
        for index, (code, language) in enumerate(LANGUAGES.items()):
            button = QPushButton(language.name)
            button.setObjectName(
                "SegFirst" if index == 0 else "SegLast" if index == last_index else "SegMid")
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            self.language_group.addButton(button)
            self.language_buttons[code] = button
            segment.addWidget(button)
        row.addLayout(segment)
        row.addStretch(1)

        self.speak_checkbox = QCheckBox("Speak replies")
        self.speak_checkbox.setChecked(True)
        self.speak_checkbox.setToolTip("Read each assistant reply aloud in the selected language")
        row.addWidget(self.speak_checkbox)

        pause_label = QLabel("Send after pause")
        pause_label.setObjectName("Field")
        row.addWidget(pause_label)
        self.delay_combo = QComboBox()
        for seconds in SEND_DELAY_CHOICES:
            self.delay_combo.addItem(f"{seconds} s", seconds)
        self.delay_combo.setToolTip("How long to wait after your last letter before the message is sent")
        row.addWidget(self.delay_combo)

        self.camera_button = QPushButton("Start camera")
        self.camera_button.setObjectName("Primary")
        self.camera_button.setMinimumWidth(120)
        row.addWidget(self.camera_button)
        return toolbar

    def _create_camera_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(12, 12, 12, 12)
        self.camera_preview = QLabel("Camera is off")
        self.camera_preview.setObjectName("CameraPreview")
        self.camera_preview.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.camera_preview.setWordWrap(True)
        self.camera_preview.setMinimumSize(480, 270)
        self.camera_preview.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        layout.addWidget(self.camera_preview, 1)
        return card

    def _create_composer_card(self) -> QFrame:
        card = self._card()
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Signed message")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.send_hint = QLabel("")
        self.send_hint.setObjectName("Small")
        header.addWidget(self.send_hint)
        layout.addLayout(header)

        self.recognized_text = QTextEdit()
        self.recognized_text.setObjectName("Composer")
        self.recognized_text.setPlaceholderText("Letters you sign appear here")
        self.recognized_text.setMinimumHeight(84)
        self.recognized_text.setMaximumHeight(120)
        layout.addWidget(self.recognized_text)

        sign_row = QHBoxLayout()
        sign_row.setSpacing(12)
        self.sign_label = QLabel("")
        self.sign_label.setObjectName("Small")
        self.sign_label.setMinimumWidth(150)
        sign_row.addWidget(self.sign_label)
        self.hold_bar = QProgressBar()
        self.hold_bar.setRange(0, 100)
        self.hold_bar.setTextVisible(False)
        self.hold_bar.setValue(0)
        sign_row.addWidget(self.hold_bar, 1)
        layout.addLayout(sign_row)

        buttons = QHBoxLayout()
        buttons.setSpacing(8)
        self.space_button = QPushButton("Space")
        self.backspace_button = QPushButton("Backspace")
        self.repeat_button = QPushButton("Repeat letter")
        self.period_button = QPushButton(".")
        self.period_button.setToolTip("End the sentence with a full stop")
        self.clear_button = QPushButton("Clear")
        for button in (self.space_button, self.backspace_button, self.repeat_button,
                       self.period_button, self.clear_button):
            buttons.addWidget(button)
        buttons.addStretch(1)
        self.send_now_button = QPushButton("Send now")
        self.send_now_button.setObjectName("Primary")
        buttons.addWidget(self.send_now_button)
        layout.addLayout(buttons)
        return card

    def _create_conversation_card(self) -> QFrame:
        card = self._card()
        card.setMinimumWidth(380)
        layout = QVBoxLayout(card)
        layout.setContentsMargins(16, 14, 16, 16)
        layout.setSpacing(10)

        header = QHBoxLayout()
        title = QLabel("Conversation")
        title.setObjectName("CardTitle")
        header.addWidget(title)
        header.addStretch(1)
        self.stop_speech_button = QPushButton("Stop speech")
        self.stop_speech_button.setObjectName("Quiet")
        self.stop_speech_button.setVisible(False)
        header.addWidget(self.stop_speech_button)
        self.new_chat_button = QPushButton("Clear conversation")
        self.new_chat_button.setObjectName("Quiet")
        header.addWidget(self.new_chat_button)
        layout.addLayout(header)

        self.key_notice = QLabel(
            "AI replies are off because the GROQ_API_KEY environment variable is not set. "
            "Set it and restart the app."
        )
        self.key_notice.setObjectName("Notice")
        self.key_notice.setWordWrap(True)
        self.key_notice.setVisible(not self.service.configured)
        layout.addWidget(self.key_notice)

        self.transcript = QTextBrowser()
        self.transcript.setObjectName("Transcript")
        self.transcript.setOpenLinks(False)
        self.transcript.setOpenExternalLinks(False)
        self.transcript.document().setDefaultStyleSheet(
            f'a {{ color: {PALETTE["ACCENT"]}; text-decoration: none; }}'
        )
        layout.addWidget(self.transcript, 1)

        row = QHBoxLayout()
        row.setSpacing(8)
        self.chat_input = QLineEdit()
        self.chat_input.setPlaceholderText("Type a message")
        row.addWidget(self.chat_input, 1)
        self.send_button = QPushButton("Send")
        self.send_button.setObjectName("Primary")
        row.addWidget(self.send_button)
        layout.addLayout(row)
        return card

    def _connect_signals(self):
        # Every visible control is connected exactly once.
        for code, button in self.language_buttons.items():
            button.clicked.connect(lambda _checked=False, c=code: self._set_language(c))
        self.speak_checkbox.toggled.connect(
            lambda checked: self.settings.setValue("speak_replies", bool(checked)))
        self.delay_combo.currentIndexChanged.connect(self._delay_changed)
        self.camera_button.clicked.connect(self.toggle_camera)

        self.recognized_text.textChanged.connect(self._recognized_text_edited)
        self.space_button.clicked.connect(lambda: self.apply_text_command("space"))
        self.backspace_button.clicked.connect(lambda: self.apply_text_command("backspace"))
        self.repeat_button.clicked.connect(lambda: self.apply_text_command("repeat"))
        self.period_button.clicked.connect(lambda: self.apply_text_command("punctuation", "."))
        self.clear_button.clicked.connect(self.clear_text)
        self.send_now_button.clicked.connect(lambda: self._send_signed(auto=False))

        self.send_button.clicked.connect(self._send_typed)
        self.chat_input.returnPressed.connect(self._send_typed)
        self.new_chat_button.clicked.connect(self.clear_conversation)
        self.stop_speech_button.clicked.connect(self.stop_speech)
        self.transcript.anchorClicked.connect(self._transcript_link)

    def _restore_settings(self):
        self.language_buttons[self.language].setChecked(True)
        speak = self.settings.value("speak_replies", True)
        self.speak_checkbox.setChecked(str(speak).lower() not in ("false", "0"))
        try:
            delay = int(self.settings.value("send_delay", DEFAULT_SEND_DELAY))
        except (TypeError, ValueError):
            delay = DEFAULT_SEND_DELAY
        if delay not in SEND_DELAY_CHOICES:
            delay = DEFAULT_SEND_DELAY
        self.delay_combo.setCurrentIndex(SEND_DELAY_CHOICES.index(delay))
        self.segmenter.send_delay = float(delay)

    # ---- settings handlers -------------------------------------------------------------

    def _set_language(self, code: str):
        if code in LANGUAGES:
            self.language = code
            self.settings.setValue("language", code)

    def _delay_changed(self, index: int):
        seconds = self.delay_combo.itemData(index)
        if seconds:
            self.segmenter.send_delay = float(seconds)
            self.settings.setValue("send_delay", int(seconds))

    # ---- camera ------------------------------------------------------------------------

    def toggle_camera(self):
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.stop_camera()
        else:
            self.start_camera()

    def start_camera(self):
        if self.camera_worker is not None and self.camera_worker.isRunning():
            return
        self._camera_error = ""
        self._hand_present = False
        self.camera_preview.setPixmap(QPixmap())
        self.camera_preview.setText("Starting camera…")
        worker = ASLCameraWorker()
        self.camera_worker = worker
        worker.frame_ready.connect(self.update_camera_frame)
        worker.text_updated.connect(self.update_recognized_text)
        worker.status_updated.connect(self.update_camera_status)
        worker.camera_error.connect(self._camera_failed)
        worker.hand_state.connect(self._hand_state_changed)
        worker.prediction.connect(self._prediction_changed)
        worker.finished.connect(lambda w=worker: self._camera_finished(w))
        worker.finished.connect(worker.deleteLater)
        self.camera_button.setText("Stop camera")
        self.camera_button.setObjectName("")
        self._restyle(self.camera_button)
        # Hand the editable text to the new worker so nothing typed is lost.
        worker.buffer.text = self.recognized_text.toPlainText()
        worker.start()

    def stop_camera(self):
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            self.camera_button.setEnabled(False)
            worker.stop()
            worker.wait(3000)

    def update_camera_status(self, message: str):
        if self.camera_worker is not None and not self._camera_error and self.camera_preview.pixmap().isNull():
            self.camera_preview.setText(message)

    def _camera_failed(self, message: str):
        self._camera_error = message
        self.camera_preview.setPixmap(QPixmap())
        self.camera_preview.setText(f"The camera could not be used\n\n{message}")

    def _camera_finished(self, worker):
        if self.camera_worker is not worker:
            return
        self.camera_worker = None
        self._hand_present = False
        self.camera_button.setEnabled(True)
        self.camera_button.setText("Start camera")
        self.camera_button.setObjectName("Primary")
        self._restyle(self.camera_button)
        self.camera_preview.setPixmap(QPixmap())
        if not self._camera_error:
            self.camera_preview.setText("Camera is off")
        self.sign_label.setText("")
        self.hold_bar.setValue(0)

    @staticmethod
    def _restyle(widget: QWidget):
        # Qt only re-reads an object-name selector after the style is re-applied.
        widget.style().unpolish(widget)
        widget.style().polish(widget)

    def update_camera_frame(self, image: QImage):
        if self.camera_worker is None:
            return
        pixmap = QPixmap.fromImage(image)
        self.camera_preview.setPixmap(
            pixmap.scaled(
                self.camera_preview.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
        )

    def _hand_state_changed(self, present: bool):
        self._hand_present = present

    def _prediction_changed(self, letter: str, confidence: float, progress: float):
        if letter:
            self.sign_label.setText(f"Detected {letter}  {confidence:.0%}")
        else:
            self.sign_label.setText("No hand detected")
        self.hold_bar.setValue(int(progress * 100))

    # ---- signed text -------------------------------------------------------------------

    def update_recognized_text(self, text: str):
        # Block the edit callback while applying text produced by the recognition worker.
        if self.recognized_text.toPlainText() != text:
            self._set_recognized_text(text)

    def _set_recognized_text(self, text: str):
        self._setting_recognized_text = True
        try:
            self.recognized_text.setPlainText(text)
            cursor = self.recognized_text.textCursor()
            cursor.movePosition(QTextCursor.MoveOperation.End)
            self.recognized_text.setTextCursor(cursor)
        finally:
            self._setting_recognized_text = False

    def _recognized_text_edited(self):
        if self._setting_recognized_text:
            return
        self._last_edit = time.monotonic()
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            worker.enqueue_command("set_text", self.recognized_text.toPlainText())

    def clear_text(self):
        self.apply_text_command("clear")

    def apply_text_command(self, command: str, value: str | None = None):
        worker = self.camera_worker
        if worker is not None and worker.isRunning():
            worker.enqueue_command(command, value)
            if command == "clear":
                self._set_recognized_text("")
            return

        current = self.recognized_text.toPlainText()
        if command == "clear":
            current = ""
        elif command == "space":
            if current and not current.endswith(" "):
                current += " "
        elif command == "backspace":
            current = current[:-1]
        elif command == "repeat":
            if current and current[-1] != " ":
                current += current[-1]
        elif command == "punctuation":
            if current and current[-1] != " ":
                current = current.rstrip(".,?!") + (value or ".")
        self._set_recognized_text(current)

    def _tick(self):
        """Watches the signed text and sends it once the signer has paused."""
        now = time.monotonic()
        worker = self.camera_worker
        running = worker is not None and worker.isRunning()
        if not running or not self.service.configured:
            self.segmenter.reset(now)
            self.send_hint.setText("")
            return

        # While the person is typing in the box, hold off so a half-written edit is not sent.
        editing = self.recognized_text.hasFocus() and now - self._last_edit < EDIT_HOLD_SECONDS
        action = self.segmenter.update(
            self.recognized_text.toPlainText(), self._hand_present, now, suspended=editing)
        remaining = self.segmenter.remaining(now)
        if remaining is not None and editing:
            self.send_hint.setText("Editing. Press Send now when ready")
        elif remaining is not None and remaining <= self.segmenter.send_delay - 1.0:
            self.send_hint.setText(f"Sending in {max(1, math.ceil(remaining))} s")
        else:
            self.send_hint.setText("")

        if action == "space":
            self.apply_text_command("space")
        elif action == "send":
            self._send_signed(auto=True)

    # ---- sending -----------------------------------------------------------------------

    def _require_service(self) -> bool:
        if self.service.configured:
            return True
        self._add_note("Set the GROQ_API_KEY environment variable and restart to send messages.", "error", once=True)
        return False

    def _send_signed(self, auto: bool):
        raw = self.recognized_text.toPlainText()
        if not raw.strip():
            return
        if not self._require_service():
            return
        job, reason = self.queue.submit(raw, self.language, "signed")
        self.segmenter.reset(time.monotonic())
        if job is None:
            if reason == "busy":
                self._add_note("Still working on earlier messages. Please wait a moment.", "info")
            elif reason == "empty" and not auto:
                self._add_note("There are no letters to send yet.", "info")
            return
        # Clear immediately so the same letters can never be submitted twice.
        self.apply_text_command("clear")
        self._set_recognized_text("")
        self.transcript_items.append({"type": "job", "job": job})
        self._refresh_transcript()
        self._pump()

    def _send_typed(self):
        text = self.chat_input.text().strip()
        if not text:
            return
        if not self._require_service():
            return
        job, reason = self.queue.submit(text, self.language, "typed")
        if job is None:
            if reason == "busy":
                self._add_note("Still working on earlier messages. Please wait a moment.", "info")
            return
        self.chat_input.clear()
        self.transcript_items.append({"type": "job", "job": job})
        self._refresh_transcript()
        self._pump()

    def _pump(self):
        """Start the next queued message if nothing is in flight."""
        if self.pipeline_worker is not None:
            return
        job = self.queue.next()
        if job is None:
            return
        self._refresh_transcript()
        worker = PipelineWorker(self.service, job, trim_history(self.history), self)
        self.pipeline_worker = worker
        worker.corrected.connect(self._job_corrected)
        worker.succeeded.connect(self._job_succeeded)
        worker.unclear.connect(self._job_unclear)
        worker.failed.connect(self._job_failed)
        worker.finished.connect(lambda w=worker: self._pipeline_finished(w))
        worker.finished.connect(worker.deleteLater)
        worker.start()

    def _find_job(self, job_id: int):
        for item in self.transcript_items:
            if item["type"] == "job" and item["job"].id == job_id:
                return item["job"]
        return None            # the conversation was cleared while this was running

    def _job_corrected(self, job_id: int, text: str):
        job = self._find_job(job_id)
        if job is not None:
            job.corrected = text
            job.state = "replying"
            self._refresh_transcript()

    def _job_succeeded(self, job_id: int, reply: str):
        job = self._find_job(job_id)
        if job is None:
            return
        job.reply, job.state = reply, "done"
        self.history.append({"role": "user", "content": job.corrected})
        self.history.append({"role": "assistant", "content": reply})
        self.history = trim_history(self.history, HISTORY_LIMIT)
        self._refresh_transcript()
        if self.speak_checkbox.isChecked():
            self._start_speech(reply, job.language)

    def _job_unclear(self, job_id: int):
        job = self._find_job(job_id)
        if job is not None:
            job.state = "unclear"
            self._refresh_transcript()

    def _job_failed(self, job_id: int, message: str):
        job = self._find_job(job_id)
        if job is not None:
            job.state, job.error = "failed", message
            self._refresh_transcript()

    def _pipeline_finished(self, worker):
        self.queue.finish(worker.job)
        if self.pipeline_worker is worker:
            self.pipeline_worker = None
        self._pump()

    def clear_conversation(self):
        self.queue.clear_pending()
        self.history = []
        self.transcript_items = []
        self._shown_notes.clear()
        self._refresh_transcript()

    # ---- transcript --------------------------------------------------------------------

    def _add_note(self, text: str, level: str = "info", once: bool = False):
        if once:
            if text in self._shown_notes:
                return
            self._shown_notes.add(text)
        self.transcript_items.append({"type": "note", "text": text, "level": level})
        self._refresh_transcript()

    def _refresh_transcript(self):
        self.transcript.setHtml(render_transcript(self.transcript_items))
        bar = self.transcript.verticalScrollBar()
        bar.setValue(bar.maximum())

    def _transcript_link(self, url):
        target = url.toString()
        kind, _, rest = target.partition(":")
        if kind == "retry":
            job = self._find_job(int(rest)) if rest.isdigit() else None
            if job is not None and self.queue.retry(job):
                self._refresh_transcript()
                self._pump()
            elif job is not None:
                self._add_note("Still working on earlier messages. Please wait a moment.", "info")
        elif kind == "speak":
            job_id, _, part = rest.partition(":")
            job = self._find_job(int(job_id)) if job_id.isdigit() else None
            if job is None:
                return
            text = job.reply if part == "reply" else job.corrected
            if text:
                self._start_speech(text, job.language)

    # ---- speech ------------------------------------------------------------------------

    def _start_speech(self, text: str, language_code: str):
        clean = text.strip()
        if not clean:
            return
        if self.speech_worker is not None and self.speech_worker.isRunning():
            # Replace the current utterance without blocking the interface.
            self._speech_next = (clean, language_code)
            self.speech_worker.stop_speech()
            return
        self._launch_speech(clean, language_code)

    def _launch_speech(self, text: str, language_code: str):
        worker = SpeechWorker(text, language_code, self)
        self.speech_worker = worker
        worker.speech_failed.connect(lambda m: self._add_note(m, "error", once=True))
        worker.voice_fallback.connect(lambda m: self._add_note(m, "info", once=True))
        worker.finished.connect(lambda w=worker: self._speech_finished(w))
        worker.finished.connect(worker.deleteLater)
        self.stop_speech_button.setVisible(True)
        worker.start()

    def stop_speech(self):
        self._speech_next = None
        if self.speech_worker is not None and self.speech_worker.isRunning():
            self.speech_worker.stop_speech()

    def _speech_finished(self, worker):
        if self.speech_worker is not worker:
            return
        self.speech_worker = None
        pending, self._speech_next = self._speech_next, None
        if pending is not None:
            self._launch_speech(*pending)
        else:
            self.stop_speech_button.setVisible(False)

    # ---- shutdown ----------------------------------------------------------------------

    def closeEvent(self, event):
        self.tick_timer.stop()
        self.queue.clear_pending()
        self._speech_next = None
        if self.camera_worker is not None and self.camera_worker.isRunning():
            self.camera_worker.stop()
            self.camera_worker.wait()
        if self.speech_worker is not None and self.speech_worker.isRunning():
            self.speech_worker.stop_speech()
            self.speech_worker.wait()
        if self.pipeline_worker is not None and self.pipeline_worker.isRunning():
            self.pipeline_worker.wait(25000)
        event.accept()


def main():
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ASLStudio()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
