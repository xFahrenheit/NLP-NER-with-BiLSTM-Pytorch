#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# In[2]:


# read the text file into a list of strings
with open('/data/train', 'r') as f:
    data = f.read().splitlines()

# split each string into three columns and store in a list
rows = []
for line in data:
    cols = line.split()
    if len(cols) == 3:
        rows.append(cols)

# create a pandas dataframe from the list of rows
df = pd.DataFrame(rows, columns = ["index", "token", "tag"])
df['index'] = df['index'].astype(int)


# In[3]:


df


# In[4]:


word2idx = {'<PAD>': 0, '<UNK>': 1}
ner2idx = {'O': 0, 'B-MISC':1, 'I-MISC':2, 'I-PER':3, 'B-LOC':4, 'I-ORG':5, 'B-PER':6, 'I-LOC':7, 'B-ORG': 8}

idx = 2  # start indexing from 2

for word in df['token']:
    if word not in word2idx:
        word2idx[word] = idx
        idx += 1


# In[5]:


# define the model architecture
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0) 
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, 128)
        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(128, output_size)


    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.classifier(lstm_out)
        
        return lstm_out


# In[6]:


# define the hyperparameters
vocab_size = len(word2idx)
embedding_dim = 100
hidden_dim = 256
output_size = len(ner2idx)
num_layers = 1
dropout = 0.33

# instantiate the model
model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout)
model.load_state_dict(torch.load('blstm1.pt', map_location ='cpu'))
model.eval()


# ## To produce dev result file for blstm1.pt:

# In[7]:


# read the text file into a list of strings
with open('/data/dev', 'r') as f:
    data = f.read().splitlines()

# split each string into three columns and store in a list
rows = []
for line in data:
    cols = line.split()
    if len(cols) == 3:
        rows.append(cols)

# create a pandas dataframe from the list of rows
df_dev = pd.DataFrame(rows, columns = ["index", "token", "tag"])
df_dev['index'] = df_dev['index'].astype(int)


# In[8]:


idx2ner = {v: k for k, v in ner2idx.items()}

# create a list to store the predicted NER tags for each word
predicted_tags = []

# initialize the encoded sentence and NER tag lists
dev_encoded_sentence = []
ner_tags = []

# loop through each row in the dataframe
for i, row in df_dev.iterrows():
    
    # check if it is the beginning of a new sentence
    if row['index'] == 1:
        
        # check if this is not the first sentence
        if i > 0:
            
            # encode the previous sentence and predict NER tags
            dev_encoded_sentence = torch.LongTensor(dev_encoded_sentence)
            with torch.no_grad():
                output = model(dev_encoded_sentence)
                predicted_tag_indices = output.argmax(dim=1)
                predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
            
        # re-initialize the encoded sentence and NER tag lists
        dev_encoded_sentence = []
        
    # encode the current word
    dev_encoded_sentence.append(word2idx.get(row['token'], word2idx['<UNK>']))
    
# encode the last sentence and predict NER tags
dev_encoded_sentence = torch.LongTensor(dev_encoded_sentence)
with torch.no_grad():
    output = model(dev_encoded_sentence)
    predicted_tag_indices = output.argmax(dim=1)
    predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
    
# add the predicted tags to the dataframe
df_dev['pred'] = predicted_tags


# In[9]:


df_dev


# In[10]:


#create dev1.out file
with open('dev1.out', 'w') as f:
    f_to_write = ""
    first_ex = True
    count = 1
    for i_row, row in df_dev.iterrows():
        if(row['index'] == 1):
            if first_ex:
                first_ex = False
            else:
                count = 1
                f_to_write += "\n"
        f_to_write += str(count) + " " + row['token'] + " " + row['pred']  + "\n"
        count+=1
    f.write(f_to_write)


# ## To produce test.out result file for blstm1.pt:

# In[12]:


# read the text file into a list of strings
with open('/data/test', 'r') as f:
    data = f.read().splitlines()

# split each string into three columns and store in a list
rows = []
for line in data:
    cols = line.split()
    if len(cols) == 2:
        rows.append(cols)

# create a pandas dataframe from the list of rows
df_test = pd.DataFrame(rows, columns = ["index", "token"])
df_test['index'] = df_test['index'].astype(int)


# In[13]:


idx2ner = {v: k for k, v in ner2idx.items()}

# create a list to store the predicted NER tags for each word
predicted_tags = []

# initialize the encoded sentence and NER tag lists
test_encoded_sentence = []
ner_tags = []

# loop through each row in the dataframe
for i, row in df_test.iterrows():
    
    # check if it is the beginning of a new sentence
    if row['index'] == 1:
        
        # check if this is not the first sentence
        if i > 0:
            
            # encode the previous sentence and predict NER tags
            test_encoded_sentence = torch.LongTensor(test_encoded_sentence)
            with torch.no_grad():
                output = model(test_encoded_sentence)
                predicted_tag_indices = output.argmax(dim=1)
                predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
            
        # re-initialize the encoded sentence and NER tag lists
        test_encoded_sentence = []
        
    # encode the current word
    test_encoded_sentence.append(word2idx.get(row['token'], word2idx['<UNK>']))
    
# encode the last sentence and predict NER tags
test_encoded_sentence = torch.LongTensor(test_encoded_sentence)
with torch.no_grad():
    output = model(test_encoded_sentence)
    predicted_tag_indices = output.argmax(dim=1)
    predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
    
# add the predicted tags to the dataframe
df_test['pred'] = predicted_tags


# In[14]:


df_test


# In[15]:


#create test1.out file
with open('test1.out', 'w') as f:
    f_to_write = ""
    first_ex = True
    count = 1
    for i_row, row in df_test.iterrows():
        if(row['index'] == 1):
            if first_ex:
                first_ex = False
            else:
                count = 1
                f_to_write += "\n"
        f_to_write += str(count) + " " + row['token'] + " " + row['pred']  + "\n"
        count+=1
    f.write(f_to_write)

