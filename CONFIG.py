import pickle
f = open("./Intermediate Files/dictionary.pkl", "rb")
note_to_int = pickle.load(f)
val_list = list(note_to_int.values())

epochs=20
batch_size=16
vocab_size=len(val_list)
input_size=20
hidden_size=50
gen_output_size=20
dis_output_size=1
n_layers=1