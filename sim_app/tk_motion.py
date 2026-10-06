"""Small, interruptible Tk transitions; all callbacks stay on the Tk thread."""

from time import monotonic
import tkinter as tk


def blend_color(start, end, progress):
    progress = max(0.0, min(1.0, progress))
    channels = [round(int(start[i:i+2], 16) * (1-progress) + int(end[i:i+2], 16) * progress)
                for i in (1, 3, 5)]
    return '#' + ''.join(f'{channel:02x}' for channel in channels)


class Transition:
    """One replaceable ease-out transition, cancelled with its owner widget."""

    def __init__(self, owner):
        self.owner = owner
        self.after_id = None
        self._update = self._complete = None
        self._closed = False
        owner.bind('<Destroy>', self._destroy, add='+')

    def start(self, update, *, duration=150, complete=None):
        self.cancel()
        if self._closed:
            return
        self._update, self._complete = update, complete
        self._started = monotonic()
        self._duration = max(1, duration) / 1000
        update(0.0)
        self.after_id = self.owner.after(16, self._step)

    def _step(self):
        self.after_id = None
        if self._closed or self._update is None:
            return
        frame_started = monotonic()
        progress = min(1.0, (frame_started - self._started) / self._duration)
        self._update(1 - (1-progress) ** 3)
        if progress >= 1:
            self._finish()
        elif self._update is not None:
            delay = max(1, round(1000 / 60 - (monotonic()-frame_started) * 1000))
            self.after_id = self.owner.after(delay, self._step)

    def _finish(self):
        complete = self._complete
        self._update = self._complete = None
        if complete is not None:
            complete()

    def finish(self):
        update, complete = self._update, self._complete
        self.cancel()
        if update is not None and not self._closed:
            update(1.0)
            if complete is not None:
                complete()

    def cancel(self):
        if self.after_id is not None:
            try:
                self.owner.after_cancel(self.after_id)
            except tk.TclError:
                pass
        self.after_id = None
        self._update = self._complete = None

    def _destroy(self, event):
        if event.widget is self.owner:
            self.cancel()
            self._closed = True
