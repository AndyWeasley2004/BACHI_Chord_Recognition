from __future__ import annotations

from itertools import combinations
from typing import List

import pandas as pd

# Supported durations and note counts (as in AugmentedNet)
available_durations = [4.0, 3.0, 2.0, 1.5, 1.0]
available_number_of_notes = [3, 4]


class TextureTemplate(object):
    def __init__(self, duration: float, notes: List[str], intervals: List[str]):
        self.duration = duration
        self.notes = notes
        self.intervals = intervals
        self.n = len(notes)
        if duration not in available_durations:
            raise ValueError("Unsupported duration")
        if self.n not in available_number_of_notes:
            raise ValueError("Unsupported number of notes")

    def triad(self) -> pd.DataFrame:
        raise NotImplementedError()

    def seventh(self) -> pd.DataFrame:
        raise NotImplementedError()

    def render(self) -> pd.DataFrame:
        if self.n == 3:
            return self.triad()
        return self.seventh()


class BlockChord(TextureTemplate):
    def triad(self) -> pd.DataFrame:
        data = {
            "s_offset": [0.0],
            "s_duration": [self.duration],
            "s_notes": [self.notes],
            "s_intervals": [[self.intervals[0], self.intervals[1]] if len(self.intervals) >= 2 else []],
            "s_isOnset": [[True, True, True]],
        }
        return pd.DataFrame(data)

    def seventh(self) -> pd.DataFrame:
        data = {
            "s_offset": [0.0],
            "s_duration": [self.duration],
            "s_notes": [self.notes],
            "s_intervals": [[self.intervals[0], self.intervals[1], self.intervals[2]] if len(self.intervals) >= 3 else []],
            "s_isOnset": [[True, True, True, True]],
        }
        return pd.DataFrame(data)


class Alberti(TextureTemplate):
    def triad(self) -> pd.DataFrame:
        dur = self.duration / 4
        rows = [
            (0.0, dur, [self.notes[0]], [], [True]),
            (dur, dur, [self.notes[2]], [], [True]),
            (2*dur, dur, [self.notes[1]], [], [True]),
            (3*dur, dur, [self.notes[2]], [], [True]),
        ]
        return pd.DataFrame(rows, columns=["s_offset", "s_duration", "s_notes", "s_intervals", "s_isOnset"])

    def seventh(self) -> pd.DataFrame:
        dur = self.duration / 4
        rows = [
            (0.0, dur, [self.notes[0]], [], [True]),
            (dur, dur, [self.notes[3]], [], [True]),
            (2*dur, dur, [self.notes[1]], [], [True]),
            (3*dur, dur, [self.notes[2]], [], [True]),
        ]
        return pd.DataFrame(rows, columns=["s_offset", "s_duration", "s_notes", "s_intervals", "s_isOnset"])


class BassSplit(TextureTemplate):
    def triad(self) -> pd.DataFrame:
        dur = self.duration / 2
        rows = [
            (0.0, dur, [self.notes[0]], [], [True]),
            (dur, dur, [self.notes[1], self.notes[2]], [self.intervals[1] if len(self.intervals)>1 else ""], [True, True]),
        ]
        return pd.DataFrame(rows, columns=["s_offset", "s_duration", "s_notes", "s_intervals", "s_isOnset"])

    def seventh(self) -> pd.DataFrame:
        dur = self.duration / 2
        rows = [
            (0.0, dur, [self.notes[0]], [], [True]),
            (dur, dur, [self.notes[1], self.notes[2], self.notes[3]], self.intervals[1:3] if len(self.intervals)>=3 else [], [True, True, True]),
        ]
        return pd.DataFrame(rows, columns=["s_offset", "s_duration", "s_notes", "s_intervals", "s_isOnset"])


available_templates = {
    "BlockChord": BlockChord,
    "Alberti": Alberti,
    "BassSplit": BassSplit,
}


def _get_intervals(notes: List[str]) -> List[str]:
    # Fallback: return placeholder intervals; exact spelling not critical for encoders
    try:
        from music21 import pitch as m21_pitch, interval as m21_interval
        root = m21_pitch.Pitch(notes[0])
        intervs = []
        for n in notes[1:]:
            intervs.append(m21_interval.Interval(root, m21_pitch.Pitch(n)).simpleName)
        return intervs
    except Exception:
        return []


def apply_texture_template(duration: float, notes: List[str], intervals: List[str], template_name: str | None = None) -> pd.DataFrame:
    if not template_name:
        # choose a default reasonable template
        template_name = "BlockChord"
    cls = available_templates.get(template_name, BlockChord)
    tmpl = cls(duration, notes, intervals or _get_intervals(notes))
    return tmpl.render()