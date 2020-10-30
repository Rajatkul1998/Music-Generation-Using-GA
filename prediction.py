from Generator import Generator
import torch
import music21
import CONFIG 
import pickle

model = Generator(CONFIG.input_size,CONFIG.hidden_size,CONFIG.gen_output_size)
model.load_state_dict(torch.load('./Intermediate Files/model.pth'))

f = open("./Intermediate Files/dictionary.pkl", "rb")
note_to_int = pickle.load(f)

key_list = list(note_to_int.keys()) 
val_list = list(note_to_int.values())


def prediction(model,noise):
  pred=model.forward(noise)
  pred=pred.reshape(-1)
  pred=pred.tolist()
  return pred


noise_float=torch.randint(low=0,high=87,size=(1,CONFIG.input_size))
noise=noise_float.long()
pred=prediction(model,noise)

def get_notes(pred,key_list,val_list):
  predicted_notes=[]
  for num in pred:
    num=int(num)  
    predicted_notes.append(key_list[num])
  return predicted_notes

predicted_notes=get_notes(pred,key_list,val_list)

s = music21.stream.Stream()
for note in predicted_notes:
    s.append(music21.note.Note(note))

s.write("midi", "./Intermediate Files/generated_music.mid")    