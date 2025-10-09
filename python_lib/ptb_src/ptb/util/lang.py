from enum import Enum
from datetime import datetime, UTC

class CommonSymbols(Enum):
    Degrees = ('°', u'\u00B0', 'DejaVuSans')
    menu = ('≡', u'\u2261', 'DejaVuSans')
    command = ('⌘', u'\u2318', 'DejaVuSans')
    crosshair = ('⌖', u'\u2316', 'DejaVuSans')
    bullseye = ('◎', u'\u25CE', 'DejaVuSans')
    camera = ('📷', u'\U0001F4F7', 'seguisym')
    save = ('💾', u'\U0001F4BE', 'seguisym')
    bars = ('📶', u"\U0001F4F6", 'seguisym')
    dot = ('●', u'\u25CF', 'seguisym')
    cogwheel = ('⚙', u'\u2699', 'seguisym')
    hammer = ('🔨', u"\U0001F528", 'seguisym')
    battery = ('🔋', u"\U0001F50B", 'seguisym')
    warning = ('⚠', u"\u26A0", 'DejaVuSans')
    flag = ('🚩', u"\U0001F6A9", 'seguisym')
    scan_icon = ('🔍', u"\U0001F50E", 'seguisym')
    set_square = ('📐', u"\U0001F4D0", 'seguisym')
    computer = ('💻', u"\U0001F4BB", 'seguisym')
    MINIDISC = ('💽', u"\U0001F4BD", "seguisym")

def milli():
    date = datetime.now(UTC).replace(tzinfo=None) - datetime(1970, 1, 1)
    seconds = (date.total_seconds())
    milliseconds = round(seconds * 1000)
    return milliseconds
