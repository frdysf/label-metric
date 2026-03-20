import itertools as it
import re
from typing import Union

def str2midi(note_string : str) -> Union[int, None]:
  """
  Given a note string name (e.g. "Bb4"), returns its MIDI pitch number.
      https://pythonhosted.org/audiolazy/_modules/audiolazy/lazy_midi.html#str2midi

  Args:
    note_string: Note string name, e.g. "Bb4".
  
  Returns:
    MIDI pitch value.
  """
  MIDI_A4 = 69
  data = note_string.strip().lower()

  if re.search(r'^[A-Ga-g]#?\d$', data) is None:
    return -1 # invalid note string, e.g. slides ("C1_G1")
 
  name2delta = {"c": -9, "d": -7, "e": -5, "f": -4, "g": -2, "a": 0, "b": 2}
  accident2delta = {"b": -1, "#": 1, "x": 2}
  accidents = list(it.takewhile(lambda el: el in accident2delta, data[1:]))
  octave_delta = int(data[len(accidents) + 1:]) - 4
  return (MIDI_A4 +
          name2delta[data[0]] + # Name
          sum(accident2delta[ac] for ac in accidents) + # Accident
          12 * octave_delta # Octave
          )
