import pyttsx3
from threading import Lock

class Speaker:
    def __init__(self, rate=180, volume=1.0):
        self.eng = pyttsx3.init()
        self.eng.setProperty('rate', rate)
        self.eng.setProperty('volume', volume)
        self.lock = Lock()

    def say(self, text: str):
        if not text: return
        with self.lock:
            self.eng.stop()
            self.eng.say(text)
            self.eng.runAndWait()
