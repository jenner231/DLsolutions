# -*- coding: utf-8 -*-
"""
Created on Mon Oct 18 17:47:38 2021

@author: au207178
"""


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import os

#https://www.kaggle.com/eswarchandt/neural-machine-translation-with-attention-dates

from faker import Faker
import random
from tqdm import tqdm
from babel.dates import format_date

from faker import Faker
fake = Faker()

Faker.seed(101)
random.seed(101)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



#%% pytorch dataset

class datesDataset(torch.utils.data.Dataset):
    def __init__(self,locale='da',inputLength=40,outputLength=12, dataSetSize=100):
        
        self.inputLength=inputLength
        self.outputLength=outputLength
        self.length=dataSetSize
        self.lan=locale
        
        self.FORMATS= ['short', # d/M/YY
           'medium', # MMM d, YYY
           'long', # MMMM dd, YYY
           'full', # EEEE, MMM dd, YYY
           'd MMM YYY', 
           'd MMMM YYY',
           'dd/MM/YYY',
           'EE d, MMM YYY',
           'EEEE d, MMMM YYY']
        

        #generate vocabularies:
        alphabet=sorted(tuple('abcdefghijklmnopqrstuvwxyzæøå'))
        numbers=sorted(tuple('0123456789'))
        symbols=['<SOS>','<EOS>',' ',',','.','/','-','<unk>', '<pad>'];
        self.humanVocab=dict(zip(symbols+numbers+alphabet,
                            list(range(len(symbols)+len(numbers)+len(alphabet)))))
        self.machineVocab =dict(zip(symbols+numbers,list(range(len(symbols)+len(numbers)))))
        self.invMachineVocab= {v: k for k, v in self.machineVocab.items()}

    def string_to_int(self,string, length, vocab):
        string = string.lower()
        

        if not len(string)+2<=length: #+2 to make room for SOS and EOS
            print(len(string),string)
            print('Length:',length)
            
            raise AssertionError()

        
        rep = list(map(lambda x: vocab.get(x, '<unk>'),string))
        rep.insert(0,vocab['<SOS>']); rep.append(vocab['<EOS>']) #add start and of sequence
        
        if len(string) < length:
            rep += [vocab['<pad>']] * (length - len(rep))
        
        return rep        
        
    def __len__(self):
        return self.length
        
    def __getitem__(self, idx):
        dt = fake.date_object()

        date = format_date(dt, format=random.choice(self.FORMATS), locale=self.lan)
        human_readable = date.lower().replace(',', '')
        machine_readable = dt.isoformat()
        
        humanEncoded=torch.LongTensor(self.string_to_int(human_readable,self.inputLength,self.humanVocab))
        machineEncoded=torch.LongTensor(self.string_to_int(machine_readable,self.outputLength,self.machineVocab))
        

        
        return human_readable, machine_readable, humanEncoded,machineEncoded

e=datesDataset()
human_readable, machine_readable, humanEncoded,machineEncoded=e[0]



