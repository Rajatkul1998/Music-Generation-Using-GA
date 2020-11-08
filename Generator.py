
import CONFIG
import torch 
import torch.nn as nn
class Generator(nn.Module):
    
    def __init__(self, input_size,hidden_size,output_size,n_layers):
        
        super(Generator, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.n_layers=n_layers
        self.embed=nn.Embedding(CONFIG.vocab_size,self.input_size)
        self.lstm=nn.LSTM(self.input_size,self.hidden_size,self.n_layers,batch_first=True)

        self.l1=nn.Linear(self.hidden_size,self.output_size)

    def init_hidden(self, batch_size):
      weight = next(self.parameters()).data
      hidden = (weight.new(self.n_layers, batch_size, self.hidden_size).zero_(),
                      weight.new(self.n_layers, batch_size, self.hidden_size).zero_())  
      return hidden
        
    def forward(self, x,hidden):
      #batch_size=x.size(0)
      embed_x=self.embed(x)
      lstm_out,(h_n,c_n)=self.lstm(embed_x,hidden)
      lstm_out = lstm_out[:,-1,:]
      f_out=self.l1(lstm_out)

      return f_out,hidden