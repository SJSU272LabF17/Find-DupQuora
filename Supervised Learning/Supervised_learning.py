
# coding: utf-8

# In[2]:

import pandas as pd 
# conda install -c anaconda gensim
import gensim
import string
import numpy as np
from sklearn import svm
from sklearn.metrics import accuracy_score


# In[3]:

# import main file into a dataframe
main_df = pd.read_csv('data/quora_duplicate_questions.tsv', sep='\t', header=0)


# In[4]:

#Load Google's pre-trained word2vec model
#https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit
model = gensim.models.KeyedVectors.load_word2vec_format('model/GoogleNews-vectors-negative300.bin', binary=True)


# In[5]:

#test our model on a random word
test = model.wv['computer']
print test


# In[6]:

#get the feature length of word
vec_len = len(test)
print vec_len


# In[7]:

"""
Function that takes a sentence as an argument, and returns a feature vector(300x1 array) out of it.
We split the sentence into individual words, apply word2vec model on each of them, and take an average of all
vectors, which represents the vector for our sentence."""

def sentence2vec(sentence):
    # clean the punctuation marks from the sentence
    clean_sentence = sentence.translate(None, string.punctuation)
    # split the senctence into a list of words
    word_list = clean_sentence.split()
    # get number of words in sentence
    sen_length = len(word_list)
    # initialize the empty array that represens the sentence vector
    sentence_vec_sum = [0]*300
    # loop through the words
    for word in word_list: 
        try: 
            # get the word vector from the pretrained model
            word_vec = model.wv[word]
            # 
            sentence_vec_sum[:] = [sentence_vec_sum[i] + word_vec[i] for i in xrange(len(sentence_vec_sum))]
        except KeyError:
            # this will happen when the word doesn't exist in the vocabulary of the original model. This typically
            # includes stopwords, and we won't include them in our average
            sen_length -= 1
    
    # since we need the average, we need to divide each element in the array by the size of the sentence
    sentence_vec = [element / sen_length for element in sentence_vec_sum]
   
    
    return sentence_vec
        


# In[8]:

str = """What is the step by step guide to invest in; share! market, in. india?"""
sentence2vec(str)


# In[9]:

# grab the first row from the dataframe 
main_df.loc[0].values[4]


# In[10]:

# write a function to parse dataframe into np array
"""Function that takes a df row as an argument and converts into two np array rows"""
def df_to_np(df_row):
    # vector from sentence 1
    try:
        vec_1 = sentence2vec(df_row[3])
    except ZeroDivisionError:
        vec_1 = [0]*300
    # vector from sentence 2
    try:
        vec_2 = sentence2vec(df_row[4])
    except ZeroDivisionError:
        vec_2 = [0]*300
    # absolute difference between the vectors
    row_vec = [abs(x-y) for x,y in zip(vec_1, vec_2)]
    # get the is_duplicate out of the array
    is_dupe = df_row[5]
    
    return row_vec, is_dupe


# In[17]:

# Initialize np arrays
data_size = 10000
X = np.zeros((data_size,300))
Y = np.zeros((data_size,1))
x = np.zeros((data_size,300))
y = np.zeros((data_size,1))

# Traverse the dataframe and add the results into the relevant numpy arrays
for i in xrange(data_size):
    row_vec, is_dupe = df_to_np(main_df.loc[i])
    X[i] = row_vec
    Y[i] = is_dupe
for i in xrange(int(0.6*data_size)):
    x[i] = X[i]
    y[i] = Y[i]
    


# In[23]:

# Time for some ML play!

# Initialize the model
clf = svm.SVC()
# Train it on the dataset
clf.fit(x, y)


# In[24]:

predicted = clf.predict(X[int(0.6*data_size):data_size][:])
y_test = Y[int(0.6*data_size):data_size]
print accuracy_score(y_test, predicted)


# In[25]:

# https://stackoverflow.com/questions/19629331/python-how-to-find-accuracy-result-in-svm-text-classifier-algorithm-for-multil


# In[ ]:



