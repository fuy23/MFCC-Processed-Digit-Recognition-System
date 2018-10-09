import pickle
import numpy as np
import cPickle
import json
import torch
import collections
from main import Net


# load the unpickle object into a variable
dict_pickle = cPickle.load(open("audio_model.pkl","r"))

if __name__ == '__main__':
    #print dict_pickle
    the_model = Net(39, 100, 10)
    the_model.load_state_dict(torch.load("demo_model.pkl"))


    x = the_model.state_dict().get('fc1.bias').numpy().reshape(1,100)
    print x.dtype
    #temp = np.zeros((3,4))
    #newFile = open('output.txt', 'w')
    #pickle.dump(the_model.state_dict().get('fc1.weight').numpy()[0], newFile)
    np.savetxt('bias.txt', x, delimiter='\n')

    #print the_model.state_dict().get('fc4.weight')
    '''text_file = open("output.txt", "w")
    text_file.write(dict_pickle['f2.weight'])
    text_file.close()'''
