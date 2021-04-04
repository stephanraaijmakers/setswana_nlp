import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE

from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf


from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse

tokenizer = AutoTokenizer.from_pretrained("./tswana_models/output")
model = AutoModel.from_pretrained("./tswana_models/output", output_attentions=True)

with open("./TSN_CORPORA/all.sentences.txt","r") as fp:
    lines=fp.readlines()

lines=[line.rstrip() for line in lines]
    
def find_similarities(model,layers=[0,1,2,3,4,5,6,10],heads=[0,1,2,5]):
    labels = lines[:100] #
#    labels=['Ka ponyo ya leitlho lefatshe la nona','Ke bereketse kwa Kanye lobaka lo lo leele']
    tokens = []

    inputs=[]
    raw_inputs=[]
    new_labels=[]
    for label in labels:
        label=' '.join(label.split(" "))
        inputs.append(tokenizer.encode(label, return_tensors='pt'))
        new_labels.append('_'.join(label))

    n=0    
    for input in inputs:
        outputs = model(input)
        attention = outputs[-1]
        nb_words=len(inputs[n][0])
        n+=1
        sum_v=np.zeros(nb_words)
        for layer in layers:
            for head in heads:
                for i in range(nb_words):
                    for j in range(nb_words):
                        try:
                            sum_v = np.add(sum_v,attention[layer][head][i][j].detach().numpy())
                        except:
                            break
        tokens.append(sum_v)

    max_len=0    
    for token in tokens:
        l=len(token)
        if l>max_len:
            max_len=l
    new_tokens=[]
    for token in tokens:
        for i in range(max_len-len(token)):
            token=np.append(token,0)
        new_tokens.append(token)

    new_tokens=np.array(new_tokens)

    A =  np.array(new_tokens)
    A_sparse = sparse.csr_matrix(A)
    similarities = cosine_similarity(A_sparse)

    for k in range(similarities.shape[0]):
        similarities[k][k]=0
        best = np.argmax(similarities[k])
        best_score=similarities[k][best]
        print("Input:%s\nBest match (%f):%s\n"%(lines[k],best_score,lines[best]))



def main():        
    find_similarities(model)

if __name__=="__main__":
    main()

