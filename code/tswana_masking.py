import matplotlib.pyplot as plt
#%matplotlib inline
import numpy as np
import re
from transformers import AutoTokenizer, AutoModel
from sklearn.manifold import TSNE
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
from transformers import pipeline
import torch
from sklearn.metrics.pairwise import cosine_similarity
from scipy import sparse
from transformers import pipeline
from transformers import AutoTokenizer, AutoModelWithLMHead


def main():
    # Example: use pre-trained model from Huggingface
    tokenizer = AutoTokenizer.from_pretrained("MoseliMotsoehli/TswanaBert")
    model = AutoModelWithLMHead.from_pretrained("MoseliMotsoehli/TswanaBert")

    # On own model: make sure you trained a model with Masking (The current Roberta model has no masking)
    #tokenizer = AutoTokenizer.from_pretrained("./tswana_models/output")
    #model = AutoModel.from_pretrained("./tswana_models/output", output_attentions=True)
    unmasker = pipeline('fill-mask', model=model,tokenizer=tokenizer)

    text='Ka ponyo ya <mask> lefatshe la nona.'
    # Ka ponyo ya leitlho lefatshe la nona
    #text = 'Ke bereketse kwa <mask> lobaka lo lo leele.'
    #'Ke bereketse kwa Kanye lobaka lo lo leele'
    print(text,unmasker(text))

if __name__=="__main__":
    main()


