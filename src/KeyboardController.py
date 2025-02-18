from ctypes import *
from time import sleep
user32 = windll.user32
kernel32 = windll.kernel32
delay = 0.01

class Key:
        a = 0x41
        b = 0x42
        c = 0x43
        d = 0x44
        e = 0x45
        f = 0x46
        g = 0x47
        h = 0x48
        i = 0x49
        j = 0x4A
        k = 0x4B
        l = 0x4C
        m = 0x4D
        n = 0x4E
        o = 0x4F
        p = 0x50
        q = 0x51
        r = 0x52
        s = 0x53
        t = 0x54
        u = 0x55
        v = 0x56
        w = 0x57
        x = 0x58
        y = 0x59
        z = 0x5A

class Keyboard:
        def __init__(self):
                Keys = Key()
        def press(key):
                """Presses key"""
                user32.keybd_event(key, 0, 0, 0)
                sleep(delay)
                user32.keybd_event(key, 0, 2, 0)
                sleep(delay)
        def hold(key):
                """Holds a key"""
                user32.keybd_event(key, 0, 0, 0)
                sleep(delay)

        def release(key):
                """Releases a key"""
                user32.keybd_event(key, 0, 2, 0)
                sleep(delay)

class Mouse:
        def __init__(self):
                left = [0x0002, 0x0004]
                right = [0x0008, 0x00010]
                middle = [0x00020, 0x00040]

        def move(x, y):
                """Moves the cursor"""
                user32.SetCursorPos(x, y)

        def click(button):
                """Clicks button"""
                user32.mouse_event(button[0], 0, 0, 0, 0)
                sleep(delay)
                user32.mouse_event(button[1], 0, 0, 0, 0)
                sleep(delay)

        def holdclick(button):
                """Start pressing button"""
                user32.mouse_event(button[0], 0, 0, 0, 0)
                sleep(delay)

        def releaseclick(button):
                """Release button"""
                user32.mouse_event(button[1])
                sleep(delay)