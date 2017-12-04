
# coding: utf-8

# ###### Importing required files 

# In[3]:


import numpy as np
import pandas as pd
import scipy.sparse as sp
from numpy.linalg import norm
from collections import Counter, defaultdict
from scipy.sparse import csr_matrix
from collections import Counter
from scipy.sparse import csr_matrix
import math 
import re


# ###### Importing and Reading data using Pandas from tsv (tab seperated values)
# In[4]:


df = pd.read_csv('quora_duplicate_questions.tsv',delimiter ='\t')
target = df['is_duplicate']
print(target.head(5))
print(df.describe())
print(df[:5])


# ###### Converting the rows of DataFrame into list which contains sentences as each instance

# In[5]:


l_1 = df['question1'].astype(str).tolist()
l_2 = df['question2'].astype(str).tolist()



# ###### Removing the grammar ex: '?' etc from the data set

# In[6]:


def noGrammer(li):
    l = li
    for i in range(len(l)):
        l[i] = re.sub('[^a-zA-Z0-9\']', ' ', l[i])
    return l


# In[7]:


list_1 = noGrammer(l_1)
list_2 = noGrammer(l_2)


# ###### Converting each sentence into list of words using split() function

# In[8]:


text = [l.split() for l in list_1]
text1 =[l.split() for l in list_2]

print (len(text), len(text1), type(text), type(text1))


# ###### Appending both the lists to create a combined csr matrix

# In[9]:


text2 = text + text1
print (len(text2))


# ###### Filtering words using minimum word length as measure

# In[10]:


#def filterLen(docs, minlen): 
#    r""" filter out terms that are too short. docs is a list of lists, each inner list is a document represented as a list of words minlen is the minimum length of the word to keep """ 
#    return [ [t for t in d if len(t) >= minlen ] for d in docs ] 
#docs_final = filterLen(text2, 2)
docs_final = text2


# ###### Building CSR(Compressed Sparse Rows) matrix where each word is represented as feature

# In[13]:


def build_matrix(docs):
    r""" Build sparse matrix from a list of documents, 
    each of which is a list of word/terms in the document.  
    """
    nrows = len(docs)
    idx = {}
    tid = 0
    nnz = 0
    for d in docs:
        nnz += len(set(d))
        for w in d:
            if w not in idx:
                idx[w] = tid
                tid += 1
    ncols = len(idx)
        
    # set up memory
    ind = np.zeros(nnz, dtype=np.int)
    val = np.zeros(nnz, dtype=np.double)
    ptr = np.zeros(nrows+1, dtype=np.int)
    i = 0  # document ID / row counter
    n = 0  # non-zero counter
    # transfer values
    for d in docs:
        cnt = Counter(d)
        keys = list(k for k,_ in cnt.most_common())
        l = len(keys)
        for j,k in enumerate(keys):
            ind[j+n] = idx[k]
            val[j+n] = cnt[k]
        ptr[i+1] = ptr[i] + l
        n += l
        i += 1
            
    mat = csr_matrix((val, ind, ptr), shape=(nrows, ncols), dtype=np.double)
    mat.sort_indices()
    
    return mat


def csr_info(mat, name="", non_empy=False):
    r""" Print out info about this CSR matrix. If non_empy, 
    report number of non-empty rows and cols as well
    """
    if non_empy:
        print("%s [nrows %d (%d non-empty), ncols %d (%d non-empty), nnz %d]" % (
                name, mat.shape[0], 
                sum(1 if mat.indptr[i+1] > mat.indptr[i] else 0 
                for i in range(mat.shape[0])), 
                mat.shape[1], len(np.unique(mat.indices)), 
                len(mat.data)))
    else:
        print( "%s [nrows %d, ncols %d, nnz %d]" % (name, 
                mat.shape[0], mat.shape[1], len(mat.data)) )


# In[14]:


mat = build_matrix(docs_final)
csr_info(mat)


# ###### Converting the CSR matrix into inverse document frequency matrix to make it easy for calculation

# In[15]:



# scale matrix and normalize its rows
def csr_idf(mat, copy=False, **kargs):
    r""" Scale a CSR matrix by idf. 
    Returns scaling factors as dict. If copy is True, 
    returns scaled matrix and scaling factors.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # document frequency
    df = defaultdict(int)
    for i in ind:
        df[i] += 1
    # inverse document frequency
    for k,v in df.items():
        df[k] = np.log(nrows / float(v))  ## df turns to idf - reusing memory
    # scale by idf
    for i in range(0, nnz):
        val[i] *= df[ind[i]]
        
    return df if copy is False else mat

def csr_l2normalize(mat, copy=False, **kargs):
    r""" Normalize the rows of a CSR matrix by their L-2 norm. 
    If copy is True, returns a copy of the normalized matrix.
    """
    if copy is True:
        mat = mat.copy()
    nrows = mat.shape[0]
    nnz = mat.nnz
    ind, val, ptr = mat.indices, mat.data, mat.indptr
    # normalize
    for i in range(nrows):
        rsum = 0.0    
        for j in range(ptr[i], ptr[i+1]):
            rsum += val[j]**2
        if rsum == 0.0:
            continue  # do not normalize empty rows
        rsum = 1.0/np.sqrt(rsum)
        for j in range(ptr[i], ptr[i+1]):
            val[j] *= rsum
            
    if copy is True:
        return mat
mat2 = csr_idf(mat, copy=True)
mat3 = csr_l2normalize(mat2, copy=True)
print("mat1:", mat[15,:20].todense(), "\n")
print("mat2:", mat2[15,:20].todense(), "\n")
print("mat3:", mat3[15,:20].todense())


# In[16]:


mat3.shape


# ###### Dividing the matrix into Query matrix 1 and Query matrix 2

# In[17]:


query_mat1 = mat3[:404290, :]
query_mat2 = mat3[404290:808580, :]


# In[18]:


query_mat2.shape


# In[19]:


query_mat1.shape


# In[20]:



rows = query_mat1.shape[0]
columns = query_mat1.shape[1]
rows
columns


# ###### Taking a sample of 405 queries to find out the cosine similarity

# In[21]:


result = []
s=0
rows1 = math.ceil(rows/1000)
column1 = columns
print(rows1, column1)
    
        


# ##### Cosine Simialrity without dimensionality reduction

# In[22]:


for i in range(rows1):
    for j in range(column1):
        s+=(query_mat1[i,j] * query_mat2[i,j])
    result.append(s)
    s=0


# ###### Sample Result of the above similarity

# In[23]:


print(result[0:10])


# In[25]:


class_test = []
for i in range(0,rows1):
    if result[i]>0.551:
        class_test.append(1) 
    else:
         class_test.append(0)
        


# In[26]:


accuracy = []
for i in range(0,rows1):
    if class_test[i] == target[i]:
        accuracy.append(1)
    else:
        accuracy.append(0)


# In[27]:


np.mean(accuracy)

