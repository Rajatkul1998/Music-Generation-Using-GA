import pickle


f = open("./Intermediate Files/dictionary.pkl", "rb")
note_to_int = pickle.load(f)


with open('./Intermediate Files/notes.bin','rb') as f1:
    notes = pickle.load(f1)

final_notes=[]
for song in notes:
    for note in song:
        final_notes.append(note)    



def convert_all_notes_int(final_notes,note_to_int):
    sequence_length=20
    network_input=[]
    for i in range(0,len(final_notes)-sequence_length,sequence_length):
        sequence_list=[]
        sequence_in = final_notes[i:i+sequence_length]
        if len(sequence_in)==sequence_length:
            for char in sequence_in:
                c=note_to_int[char]
                sequence_list.append(c)
            network_input.append(sequence_list)  
    return network_input 


network_input=convert_all_notes_int(final_notes,note_to_int)
print("ToTal Sequences:",len(network_input))
with open('./Intermediate Files/Converted_notes_to_int.bin','wb') as f:
    pickle.dump(network_input,f)