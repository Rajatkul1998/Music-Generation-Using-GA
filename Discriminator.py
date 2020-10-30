import CONFIG 
import torch
import torch.nn as nn
class Discriminator(nn.Module):
    
    def __init__(self, input_size, hidden_size,output_size):
        super(Discriminator, self).__init__()
        self.input_size=input_size
        self.hidden_size=hidden_size
        self.output_size=output_size
        self.embed=nn.Embedding(CONFIG.vocab_size,self.input_size)
        self.lstm=nn.LSTM(self.input_size,self.hidden_size,batch_first=True)
        self.l1=nn.Linear(self.hidden_size,self.output_size)
        self.prob=nn.Sigmoid()

    def init_hidden(self, batch_size):
      return(torch.autograd.Variable(torch.randn(1, batch_size, self.hidden_size)),torch.autograd.Variable(torch.randn(1, batch_size, self.hidden_size)))   
        
    def forward(self, x):
      self.hidden = self.init_hidden(x.size(0))
      embed_x=self.embed(x)
      embed_x=embed_x.reshape(x.size(0),self.input_size,-1)
      out,(ht,ct)=self.lstm(embed_x,self.hidden)
      ht=torch.squeeze(ht)
      output1=self.l1(ht)
      prob=self.prob(output1)
      return prob