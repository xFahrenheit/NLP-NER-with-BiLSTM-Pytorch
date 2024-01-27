#!/usr/bin/env python
# coding: utf-8

# In[1]:


#import necessary libraries 
import pandas as pd
import numpy as np
import torch
import torch.nn as nn


# ## load the train dataset and the model

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


# create mappings from words and NER tags to indices
ner2idx = {'O': 0, 'B-MISC':1, 'I-MISC':2, 'I-PER':3, 'B-LOC':4, 'I-ORG':5, 'B-PER':6, 'I-LOC':7, 'B-ORG': 8}
word2idx = {'<PAD>': 0, 'unk': 1}

idx = 2  # start indexing from 2

#create a dictionary of unique words and indexes
for word in df['token']:
    if word not in word2idx: 
        word2idx[word] = idx
        idx += 1

word_list = list(word2idx.keys())


# In[4]:


# load the glove embeddings
glove_embeddings = {}
with open('glove.6B.100d', encoding='utf-8') as f:
    for line in f:
        values = line.split()
        word = values[0]
        embedding = np.asarray(values[1:], dtype='float32')
        glove_embeddings[word] = embedding


# In[5]:


word_embed_matrix = []

random_array = np.concatenate((np.random.uniform(-1, 0, size=50), np.random.uniform(0, 1, size=50))) * np.random.choice([-1, 1], size=100)

def is_capitalized(word):
    return word[0].istitle()

for word in word_list:
    # check if the word is capitalized or not
    if is_capitalized(word):
        # if the word is capitalized, convert it to lower case and check if it exists in the glove embeddings
        if word.lower() in glove_embeddings:
            # if the lowercased word exists in the glove embeddings, get its embedding and concatenate a 1 to it
            word_embedding = np.concatenate([glove_embeddings[word.lower()], np.asarray([1])], axis=0)
        else:
            # if the lowercased word does not exist in the glove embeddings, get the embedding of the <unk> token and concatenate a 1 to it
            word_embedding = np.concatenate([random_array, np.asarray([1])], axis=0)
    else:
        # if the word is not capitalized, get its lowercased embedding from the glove embeddings (if it exists there) and concatenate a 0 to it
        if word.lower() in glove_embeddings:
            word_embedding = np.concatenate([glove_embeddings[word.lower()], np.asarray([0])], axis=0)
        else:
            word_embedding = np.concatenate([random_array, np.asarray([0])], axis=0) #random_array

    word_embed_matrix.append(word_embedding)


# In[6]:


from pandas.core.common import random_state

# define the model architecture
class BiLSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout):
        super(BiLSTM, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(torch.FloatTensor(word_embed_matrix), freeze=False, padding_idx=0)
        self.embedding_dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True, num_layers=num_layers, bidirectional=True)
        self.linear = nn.Linear(hidden_dim * 2, 128)
        self.linear_dropout = nn.Dropout(0.2)
        self.dropout = nn.Dropout(p = dropout)
        self.activation = nn.ELU()
        self.classifier = nn.Linear(128, output_size)
        
    def forward(self, x):
        lstm_out = self.embedding(x) 
        lstm_out = self.embedding_dropout(lstm_out) 
        lstm_out, _ = self.lstm(lstm_out) 
        lstm_out = self.dropout(lstm_out)
        lstm_out = self.linear(lstm_out)
        lstm_out = self.linear_dropout(lstm_out)
        lstm_out = self.activation(lstm_out)
        lstm_out = self.classifier(lstm_out)
        
        return lstm_out


# In[8]:


output_size = len(ner2idx)
vocab_size = len(word2idx)
embedding_dim = 101
hidden_dim = 256
num_layers = 1
dropout = 0.33

model = BiLSTM(vocab_size, embedding_dim, hidden_dim, output_size, num_layers, dropout)
model.load_state_dict(torch.load('blstm2.pt', map_location ='cpu')) #load model
model.eval()


# ## To produce test result file on blstm2.pt:

# In[9]:


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


# In[10]:


df_test


# In[11]:


idx2ner = {v: k for k, v in ner2idx.items()}

# create a list to store the predicted NER tags for each word
predicted_tags = []

# initialize the encoded sentence and NER tag lists
test_encoded_sentence = []

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
    if row['token'].isupper() and row['token'].title() in word2idx:
        test_encoded_sentence.append(word2idx[row['token'].title()])
        
    #if word exists in vocab, get the idx of the word    
    elif row['token'] in word2idx:
        test_encoded_sentence.append(word2idx[row['token']])
    
    #else assign index of unk
    else:
        test_encoded_sentence.append(word2idx['unk'])
    
# encode the last sentence and predict NER tags
test_encoded_sentence = torch.LongTensor(test_encoded_sentence)

with torch.no_grad():
    output = model(test_encoded_sentence)
    predicted_tag_indices = output.argmax(dim=1)
    predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
    
# add the predicted tags to the dataframe
df_test['pred'] = predicted_tags


# In[12]:


with open('test2.out', 'w') as f:
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


# In[13]:


df_test


# ## To produce dev result file for blstm2.pt:

# In[15]:


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


# In[16]:


idx2ner = {v: k for k, v in ner2idx.items()}

# create a list to store the predicted NER tags for each word
predicted_tags = []

# initialize the encoded sentence and NER tag lists
dev_encoded_sentence = []

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
        
    # if word is in all caps and if word has first letter as capital, consider the word in its as first letter capital form
    if row['token'].isupper() and row['token'].title() in word2idx:
        dev_encoded_sentence.append(word2idx[row['token'].title()])
    
    #if word exists in vocab, get the idx of the word 
    elif row['token'] in word2idx:
        dev_encoded_sentence.append(word2idx[row['token']])
    
    #else assign index of unk
    else:
        dev_encoded_sentence.append(word2idx['unk'])
    
# encode the last sentence and predict NER tags
dev_encoded_sentence = torch.LongTensor(dev_encoded_sentence)

with torch.no_grad():
    output = model(dev_encoded_sentence)
    predicted_tag_indices = output.argmax(dim=1)
    predicted_tags.extend([idx2ner[idx.item()] for idx in predicted_tag_indices])
    
# add the predicted tags to the dataframe
df_dev['pred'] = predicted_tags


# In[22]:


with open('dev2.out', 'w') as f:
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


# In[23]:


df_dev

