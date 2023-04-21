# -*- coding: utf-8 -*-
"""
Created on Tue Apr 18 20:07:08 2023

@author: dcb21
"""

import openai
import tiktoken
import sklearn
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from dotenv import load_dotenv
import os
from itertools import islice
import numpy as np
from utils import get_embedding, text_from_file, text_to_file, chunked_tokens, len_safe_get_embedding, num_tokens
import pdfplumber
import re
import utils
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

#print(len(get_embedding('hello world')))

#get text from each pdf

'''#alright, what are you trying to do? 

You want to see how good the embeddings are at figuring out what cases should
should be cited. So what you really should do is remove the case citations.

But that's easy in a sense because the cases will be cited throughout. I guess 
what you should do is look at the statement of facts, and see if it can guess
the cases cited, since that won't have any case references in it.

So the next thing to do is to extract statement of facts.
 '''


folder_path = 'pdfs/oad/'
            
def get_pdf_text(file_path):
    
    if os.path.isfile(file_path):
        print(f'Processing file: {file_path}')
        with pdfplumber.open(file_path) as pdf:
            text = ''
            for page in pdf.pages:
                text += page.extract_text()
        
        return text

def get_sof(text):
    
    pattern = re.compile(r'STATEMENT OF FACTS(.*?)ARGUMENT', re.DOTALL | re.IGNORECASE)
    matches = list(re.finditer(pattern, text))
    sof = matches[1].group(1)
    return sof
    
def get_authorities(text):
    
    pattern = re.compile(r'TABLE OF AUTHORITIES(.*?)Supreme', re.DOTALL | re.IGNORECASE)
    matches = list(re.finditer(pattern, text))
    auths = matches[0].group(1)
    return auths

def get_sof_embedding_rows(name, sof, average = True):
    #if average is false there will be more than one embedding vector
    #so we want to create a row with a name and index so we know what section/chunk
    #we are referring to
    
    #this is returning a ragged array, you may want to make it not ragged.
    #But we'll leave it like this for now
    try:
        embeddings = len_safe_get_embedding(sof, average=average)
        embeddings = [ [name, index, embedding] for index,embedding 
                      in enumerate(embeddings)]
    except Exception as e:
        print(e)
    return embeddings
    

def get_all_sof_embeddings(average = True):
    
    sofs = []
    for file_name in os.listdir(folder_path):
        file_path = os.path.join(folder_path, file_name)
        text = get_pdf_text(file_path)
        try:
            sof = get_sof(text)
            name_pattern = re.compile(r'[A-Z][a-z]+_[A-Z][a-z]+')
            name = re.findall(name_pattern, file_name)[0]
            sofs.append(get_sof_embedding_rows(name, sof, average))            
        except Exception as e:
            print(e)
        
    return sofs


#okay so if there are n chunks, it should come out as a list of embeddings of
#with shape n, 1536

#so what I want to do is, get all cases and authorities that are cited.
        
'''
Here's how you make the matrix of cosine similarities as ChatGPT explains
cosine similarity is calculated the same as openai does, but sklearn is 
capable of doing it with vectors and oututting a matrix.

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

# Example vectors
vector_names = ['vector_a', 'vector_b', 'vector_c']
vectors = [
    np.array([1, 2, 3]),
    np.array([4, 5, 6]),
    np.array([7, 8, 9]),
]

# Calculate the cosine similarity matrix
similarity_matrix = cosine_similarity(vectors)

# Create a DataFrame with labeled rows and columns
df = pd.DataFrame(similarity_matrix, index=vector_names, columns=vector_names)

# Print the DataFrame
print(df)



'''
        
    
    
    
    
    








# #didn't finish writing this.
# def get_oad_text():
#     for filename in os.listdir(folder_path):
#         file_path = os.path.join(folder_path, filename)






