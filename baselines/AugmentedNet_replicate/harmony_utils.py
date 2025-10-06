import os, re
from pathlib import Path
from typing import List, Tuple, Optional
from music21 import roman, converter, key
from tqdm import tqdm
from .data_utils import (
    PREFERRED_PITCH_CLASSES,
)
import pandas as pd
import re
from fractions import Fraction

class HarmonyEvent:
    def __init__(self, beat: float, rn: roman.RomanNumeral, measure: Optional[int] = None, duration: Optional[float] = None):
        if type(beat) != float:
            raise ValueError(f"beat must be a float, got {type(beat)}")
        self.abs_beat = beat
        self.rn = rn
        self.figure_string = rn.figure
        self.root_pc = rn.root().pitchClass
        self.bass_pc = rn.bass().pitchClass
        self.root = PREFERRED_PITCH_CLASSES[self.root_pc]
        self.bass = PREFERRED_PITCH_CLASSES[self.bass_pc]
        
        # Optional metadata
        self.measure = measure if measure is not None else 0
        self.duration = duration
        try:
            self.local_key = rn.key.tonicPitchNameWithCase
        except Exception:
            self.local_key = "C"
        try:
            secondaryKey = rn.secondaryRomanNumeralKey
            self.tonicized_key = secondaryKey.tonicPitchNameWithCase if secondaryKey else self.local_key
        except Exception:
            self.tonicized_key = self.local_key
        
        pitches = {p.pitchClass for p in rn.pitches}
        self.degrees = sorted(((pc - self.root_pc) % 12) for pc in pitches)
        self.quality = self._get_quality_from_degrees(self.degrees)
    
    def _strip_nonfunctional_additions(self, deg_set: set) -> set:
        """
        Return a copy of *deg_set* with non‑functional add‑intervals removed.

        ▸ “Add” tones are 2 (9th), 5 (4th/11th) and 9 (6th/13th).
        ▸ We keep them **only** when:
              • they substitute for the 3rd (→ sus2 / sus4);
              • they turn a dim triad into a full °7 (0‑3‑6‑9).
        """
        out = set(deg_set)          # work on a copy

        has_third = 3 in out or 4 in out          # minor or major 3rd present
        is_dim_triad = 3 in out and 6 in out and 4 not in out

        # 2 or 5 are colour‑tones if a third is already present
        if has_third:
            out.discard(2)          # add2 / add9
            out.discard(5)          # add4 / add11

        # 9 is a colour‑tone on *non‑diminished* triads
        if 9 in out and not is_dim_triad:
            out.discard(9)          # add6 / add13

        return out

    def _get_quality_from_degrees(self, degrees: tuple) -> str:
        """Return a lead‑sheet style quality symbol (M, m7, D7, o7 …)."""
        deg_set = set(degrees)

        # 3. build triad quality
        deg_set = self._strip_nonfunctional_additions(deg_set)
        # print(f"stripped : {deg_set}")          # (optional) debug
        triad = None
        if 4 in deg_set and 8 in deg_set:     # major + #5
            triad = '+'
        elif 3 in deg_set and 6 in deg_set and 7 not in deg_set:   # minor + b5
            triad = 'o'
        elif 4 in deg_set:                                         # plain major
            triad = 'M'
        elif 3 in deg_set:                                         # plain minor
            triad = 'm'

        if triad is None:
            return 'other'     # unrecognised sonority

        # 4. add 7th information, if any
        has_m7  = 10 in deg_set
        has_M7  = 11 in deg_set
        has_d7  =  9 in deg_set   # diminished 7th (= bb7)

        if not (has_m7 or has_M7 or has_d7):
            return triad                       # simple triad

        if triad == 'M':
            if has_m7:  return 'D7'            # dominant 7th
            if has_M7:  return 'M7'            # major‑maj7
        elif triad == 'm':
            if has_m7:  return 'm7'            # minor 7th
            if has_M7:  return 'mM7'
        elif triad == '+':
            if has_m7 or has_M7:  return '+7'
        elif triad == 'o':
            if has_d7:  return 'o7'            # fully dim
            if has_m7:  return '/o7'           # half‑dim

        return 'other' 
        
    @property
    def label(self):
        return [self.root, self.quality, self.bass]

# class HarmonyEvent:
#     def __init__(self, beat: float, rn: roman.RomanNumeral):
#         self.abs_beat = beat
#         self.figure_string = rn.figure
#         self.root_pc = rn.root().pitchClass
#         self.bass_pc = rn.bass().pitchClass
#         pitches = {p.pitchClass for p in rn.pitches}
#         self.degrees = sorted(((pc - self.root_pc) % 12) for pc in pitches)
#         self.chord_parts = self._get_chord_decomposition(self.degrees)


#     def _pc_to_name(self, pc: int) -> str:
#         """Convert a pitch‑class (0‒11) to the canonical chord‑slot name."""
#         names = [
#             "C", "C#", "D", "D#", "E", "F",
#             "F#", "G", "G#", "A", "A#", "B",
#         ]
#         return names[pc % 12]


#     def _get_chord_decomposition(self, degrees: tuple) -> list:
#         """
#         Return an ordered list
#         [root, triad, bass, seventh, ninth, eleventh, thirteenth]
#         using the slot vocab we discussed.  Bass is *always* emitted
#         even when it equals the root.
#         """
#         pcs = set(degrees) # O(1) search time

#         # --- root & bass --------------------------------------------------
#         root = self._pc_to_name(self.root_pc)
#         bass = self._pc_to_name(self.bass_pc)

#         # --- triad detection ---------------------------------------------
#         triad = "N"
#         consumed = set()
#         if 4 in pcs and 7 in pcs:  # major
#             triad = "maj"; consumed |= {4, 7}
#         elif 3 in pcs and 7 in pcs:  # minor
#             triad = "min"; consumed |= {3, 7}
#         elif 3 in pcs and 6 in pcs:  # diminished
#             triad = "dim"; consumed |= {3, 6}
#         elif 4 in pcs and 8 in pcs:  # augmented
#             triad = "aug"; consumed |= {4, 8}
#         elif 5 in pcs and 3 not in pcs and 4 not in pcs:      # sus4
#             triad = "sus4"
#             consumed |= {5, 7}       # 7 = perfect 5th of the sus-triad
#         elif 2 in pcs and 3 not in pcs and 4 not in pcs:      # sus2
#             triad = "sus2"
#             consumed |= {2, 7}       # 7 = perfect 5th of the sus-triad

#         remaining = pcs - consumed

#         if 11 in remaining:
#             seventh = "7"; remaining.remove(11)
#         elif 10 in remaining:
#             seventh = "b7"; remaining.remove(10)
#         elif 9 in remaining and triad == "dim":
#             seventh = "bb7"; remaining.remove(9)
#         else:
#             seventh = "N"

#         if 2 in remaining:
#             ninth = "9"; remaining.remove(2)
#         elif 1 in remaining:
#             ninth = "b9"; remaining.remove(1)
#         elif 3 in remaining:
#             ninth = "#9"; remaining.remove(3)
#         else:
#             ninth = "N"

#         if 5 in remaining:
#             eleventh = "11"; remaining.remove(5)
#         elif 6 in remaining and triad != "dim":   # pc 6 is #11 only outside dim triads
#             eleventh = "#11"; remaining.remove(6)
#         else:
#             eleventh = "N"

#         if 9 in remaining:                 # already solved = 13 vs bb7
#             thirteenth = "13"
#         elif 8 in remaining:
#             if triad == "aug":             # part of the triad: interpret as #5
#                 # nothing to do – pc 8 was consumed earlier
#                 thirteenth = "N"
#             else:
#                 thirteenth = "b13"
#         else:
#             thirteenth = "N"

#         return [root, triad, bass, seventh, ninth, eleventh, thirteenth]

    
#     @property
#     def label(self):
#         return self.chord_parts

def parse_rntxt_m21(
    path: Path,
) -> Tuple[List[HarmonyEvent], Tuple[int, int], str]:
    try:
        sc = converter.parse(str(path), format="romanText")

        events = []
        for r in sc.recurse().getElementsByClass("RomanNumeral"):
            # r.figure = re.sub(r'\[no\d+\]', '', r.figure) # remove noX in figure
            beat = r.getOffsetBySite(sc.flat)
            float_beat = beat if isinstance(beat, float) else float(Fraction(beat))
            try:
                measure_num = int(getattr(r, "measureNumber", 0) or 0)
            except Exception:
                measure_num = 0
            try:
                ql = float(getattr(r, "quarterLength", 0.0) or 0.0)
            except Exception:
                ql = 0.0
            events.append(HarmonyEvent(float_beat, r, measure=measure_num, duration=ql))
        events.sort(key=lambda e: e.abs_beat)
        return events
    except:
        print(f"Error parsing {path}")
        return None


def parse_dcml(path: Path) -> Tuple[List[HarmonyEvent], Tuple[int, int], str]:
    df = pd.read_csv(path, sep="\t")

    # Determine global key and mode
    try:
        gk = str(df["globalkey"].iloc[0]).strip()
    except Exception:
        gk = "C"
    try:
        is_global_minor = bool(int(df.get("globalkey_is_minor", pd.Series([0])).iloc[0]))
    except Exception:
        is_global_minor = gk.islower()
    try:
        global_key = key.Key(gk, "minor" if is_global_minor else "major")
    except Exception:
        global_key = key.Key(gk)

    events: List[HarmonyEvent] = []
    try:
        for _, row in df.iterrows():
            if pd.isna(row.get("chord", None)) or row.get("numeral", "") == "@none" or pd.isna(row.get("quarterbeats", None)):
                continue

            # Offsets and duration
            beat = float(Fraction(row["quarterbeats"]))
            duration_qb = row.get("duration_qb", None)
            try:
                duration = float(Fraction(duration_qb)) if pd.notna(duration_qb) else None
            except Exception:
                duration = None

            # Measure number (prefer consolidated measure count if present)
            measure = 0
            try:
                if "mc" in df.columns and pd.notna(row.get("mc", None)):
                    measure = int(row["mc"])  # consolidated measure count
                elif "mn" in df.columns and pd.notna(row.get("mn", None)):
                    measure = int(row["mn"])  # edition measure number
            except Exception:
                measure = 0

            # Local key from relative RN and global key
            localkey_rn = str(row.get("localkey", "i"))
            try:
                local_tonic = roman.RomanNumeral(localkey_rn, global_key).root().name
            except Exception:
                local_tonic = global_key.tonic.name
            try:
                is_local_minor = bool(int(row.get("localkey_is_minor", 0)))
            except Exception:
                is_local_minor = localkey_rn.islower()
            try:
                local_key = key.Key(local_tonic, "minor" if is_local_minor else "major")
            except Exception:
                local_key = key.Key(local_tonic)

            # RN figure assembly
            figure_string = str(row.get("numeral", "")).replace("#", "")
            form = row.get("form", None)
            if pd.notna(form):
                figure_string += str(form).replace("%", "ø")
            figbass = row.get("figbass", None)
            if pd.notna(figbass):
                try:
                    figure_string += str(int(float(figbass)))
                except Exception:
                    figure_string += str(figbass)
            changes = row.get("changes", None)
            if pd.notna(changes):
                figure_string += "".join(
                    f"[add{m}]" for m in re.findall(r"[#b]?(?:11|13|[24679])", str(changes))
                )
            relroot = row.get("relativeroot", None)
            if pd.notna(relroot):
                figure_string += "/" + str(relroot).split("/")[-1]

            rn_obj = roman.RomanNumeral(figure_string, local_key)
            events.append(HarmonyEvent(beat, rn_obj, measure=measure, duration=duration))
        events.sort(key=lambda e: e.abs_beat)
        return events
    except Exception as e:
        print(f"Error parsing {path}: {e}")
        return None