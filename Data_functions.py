from torch.autograd import Variable
import torch
import CONFIG


    
def make_some_noise(size):
    rand_tensor=torch.randint(low=0,high=87,size=(size,CONFIG.input_size))
    return rand_tensor

def real_data_target(size):
    data = Variable(torch.ones(size, 1)*0.9)
    return data

def fake_data_target(size):
    data = Variable(torch.zeros(size, 1)*0.1)
    return data    
