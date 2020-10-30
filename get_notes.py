import glob
import os
import pickle
import matplotlib.pyplot as plt
import music21
from music21 import converter, instrument, note, chord
f_names =[]
file_path = file_path = glob.glob("./Music Files/***.mid")
for file in file_path:
  file_name = os.path.basename(file)
  f_names.append(file_name)
print(f_names)

def midi_notes(filename):
   midi = converter.parse(filename)
   notes_to_parse = None
   notes=[]
   parts = instrument.partitionByInstrument(midi)
   #print(parts)
   if parts:
     notes_to_parse = parts.parts[0].recurse() # file has instrument parts
   else:
     notes_to_parse = midi.flat.notes # file has notes in a flat structure
   for element in notes_to_parse:
     if isinstance(element, note.Note):
       notes.append(str(element.pitch))
   return notes
main_list=[]
for midiname in f_names:
      path='./Music Files/'
      path+=midiname
      notes=midi_notes(path)
      main_list.append(notes)
print(main_list)

with open('./Intermediate Files/notes.bin','wb') as f:
    pickle.dump(main_list,f)