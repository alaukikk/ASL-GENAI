"""A minimal stand-in for PySide6, used only by the tests.

It lets tests/test_studio_flow.py import and drive asl_studio.py on a machine with no Qt:
signals really connect and emit, QThread.start() runs the thread body synchronously, and
widgets remember their text/state. It draws nothing, so it can verify the application's
wiring and logic but says nothing about how the interface looks.
"""

from __future__ import annotations

import sys
import types


class _Any:
    """Absorbs any attribute access or call (layout methods, enums, styling...)."""

    def __getattr__(self, name):
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()

    def __call__(self, *args, **kwargs):
        return _Any()


class _Meta(type):
    def __getattr__(cls, name):          # Qt.AlignmentFlag.AlignCenter and friends
        if name.startswith("__"):
            raise AttributeError(name)
        return _Any()


class _Bound:
    def __init__(self):
        self.slots = []

    def connect(self, slot):
        self.slots.append(slot)

    def disconnect(self, *_):
        self.slots.clear()

    def emit(self, *args):
        for slot in list(self.slots):
            slot(*args)


class Signal:
    def __init__(self, *types_):
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner):
        if instance is None:
            return self
        return instance.__dict__.setdefault(f"_signal_{self.name}", _Bound())


class QObject(metaclass=_Meta):
    def __init__(self, *args, **kwargs):
        self._text = args[0] if args and isinstance(args[0], str) else ""
        self._html = ""
        self._enabled = True
        self._visible = True
        self._checked = False
        self._focus = False
        self._object_name = ""
        self._pixmap = None
        self._items = []
        self._index = -1

    def __getattr__(self, name):
        if name.startswith("_") or name in ("slots",):
            raise AttributeError(name)
        return _Any()

    # state the application actually reads back
    def setObjectName(self, name): self._object_name = name
    def objectName(self): return self._object_name
    def setEnabled(self, value): self._enabled = bool(value)
    def isEnabled(self): return self._enabled
    def setVisible(self, value): self._visible = bool(value)
    def isVisible(self): return self._visible
    def setText(self, text): self._text = text
    def text(self): return self._text
    def hasFocus(self): return self._focus
    def deleteLater(self): pass


class QThread(QObject):
    finished = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__()
        self._running_now = False

    def start(self):
        self._running_now = True
        try:
            self.run()
        finally:
            self._running_now = False
        self.finished.emit()

    def isRunning(self): return self._running_now
    def wait(self, *_): return True
    def msleep(self, *_): pass


class QTimer(QObject):
    timeout = Signal()
    def start(self, *_): pass
    def stop(self): pass


class QSettings(QObject):
    store: dict = {}

    def __init__(self, *args, **kwargs):
        super().__init__()

    def value(self, key, default=None): return QSettings.store.get(key, default)
    def setValue(self, key, value): QSettings.store[key] = value


class Qt(metaclass=_Meta):
    pass


class QImage(QObject):
    def copy(self): return self


class QTextCursor(QObject):
    pass


class QPixmap(QObject):
    def __init__(self, *args, **kwargs):
        super().__init__()
        self._null = not args or not isinstance(args[0], QImage)

    @classmethod
    def fromImage(cls, image): return cls(image)
    def isNull(self): return self._null
    def scaled(self, *args, **kwargs): return self


class QWidget(QObject): pass
class QFrame(QWidget): pass
class QMainWindow(QWidget): pass
class QVBoxLayout(QObject): pass
class QHBoxLayout(QObject): pass
class QSizePolicy(QObject): pass
class QProgressBar(QWidget):
    def setValue(self, value): self._value = value
    def value(self): return getattr(self, "_value", 0)


class QApplication(QObject): pass


class QLabel(QWidget):
    def setPixmap(self, pixmap): self._pixmap = pixmap
    def pixmap(self): return self._pixmap if self._pixmap is not None else QPixmap()


class QPushButton(QWidget):
    clicked = Signal()
    def setCheckable(self, value): pass
    def setChecked(self, value): self._checked = bool(value)
    def isChecked(self): return self._checked
    def click(self): self.clicked.emit(False)


class QButtonGroup(QObject):
    buttonClicked = Signal()


class QCheckBox(QWidget):
    toggled = Signal()
    def setChecked(self, value):
        value = bool(value)
        changed = value != self._checked
        self._checked = value
        if changed:
            self.toggled.emit(value)
    def isChecked(self): return self._checked


class QComboBox(QWidget):
    currentIndexChanged = Signal()
    def addItem(self, text, data=None): self._items.append((text, data))
    def itemData(self, index): return self._items[index][1]
    def currentIndex(self): return self._index
    def setCurrentIndex(self, index):
        changed = index != self._index
        self._index = index
        if changed:
            self.currentIndexChanged.emit(index)


class QLineEdit(QWidget):
    returnPressed = Signal()
    textChanged = Signal()
    def clear(self): self._text = ""


class QTextEdit(QWidget):
    textChanged = Signal()
    def setPlainText(self, text):
        self._text = text
        self.textChanged.emit()
    def toPlainText(self): return self._text
    def setHtml(self, html): self._html = html
    def toHtml(self): return self._html
    def clear(self): self.setPlainText("")


class QTextBrowser(QTextEdit):
    anchorClicked = Signal()


def install():
    """Register the stub as PySide6 (a no-op if real PySide6 is importable)."""
    QSettings.store = {}
    if "PySide6" in sys.modules and not getattr(sys.modules["PySide6"], "_is_stub", False):
        return False
    package = types.ModuleType("PySide6")
    package._is_stub = True
    modules = {
        "QtCore": ["Qt", "QThread", "QTimer", "QSettings", "Signal", "QObject"],
        "QtGui": ["QImage", "QPixmap", "QTextCursor"],
        "QtWidgets": ["QApplication", "QButtonGroup", "QCheckBox", "QComboBox", "QFrame",
                      "QHBoxLayout", "QLabel", "QLineEdit", "QMainWindow", "QProgressBar",
                      "QPushButton", "QSizePolicy", "QTextBrowser", "QTextEdit",
                      "QVBoxLayout", "QWidget"],
    }
    for name, members in modules.items():
        module = types.ModuleType(f"PySide6.{name}")
        for member in members:
            setattr(module, member, globals()[member])
        setattr(package, name, module)
        sys.modules[f"PySide6.{name}"] = module
    sys.modules["PySide6"] = package
    return True
