import pickle
f = open("./Intermediate Files/dictionary.pkl", "rb")
note_to_int = pickle.load(f)
val_list = list(note_to_int.values())

epochs=20
batch_size=3
vocab_size=len(val_list)
input_size=10
hidden_size=50
gen_output_size=10
dis_output_size=1