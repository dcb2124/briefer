# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import openai
from openai.embeddings_utils import cosine_similarity
import tiktoken
from tenacity import retry, wait_random_exponential, stop_after_attempt, retry_if_not_exception_type
from dotenv import load_dotenv
import os
from itertools import islice
import numpy as np


load_dotenv()
openai.api_key = os.environ['OPENAI_API_KEY']

def num_tokens(string, encoding_name = 'cl100k_base'):
    """Returns the number of tokens in a text string."""
    encoding = tiktoken.get_encoding(encoding_name)
    num_tokens = len(encoding.encode(string))
    return num_tokens


EMBEDDING_MODEL = 'text-embedding-ada-002'
EMBEDDING_CTX_LENGTH = 8191
EMBEDDING_ENCODING = 'cl100k_base'

# let's make sure to not retry on an invalid request, because that is what we want to demonstrate
@retry(wait=wait_random_exponential(min=1, max=20), stop=stop_after_attempt(6), 
       retry=retry_if_not_exception_type(openai.InvalidRequestError))
def get_embedding(text_or_tokens, model=EMBEDDING_MODEL):
    #returns list of values, not json response
    return openai.Embedding.create(input=text_or_tokens, model=model)["data"][0]["embedding"]


def text_from_file(file):
    
    with(open(file, 'r') as f):
        txt = f.read()
        
    return txt

def text_to_file(text, file_path):
    with open(file_path, 'w') as f:
        f.write(file_path)

def batched(iterable, n):
    """Batch data into tuples of length n. The last batch may be shorter."""
    # batched('ABCDEFG', 3) --> ABC DEF G
    if n < 1:
        raise ValueError('n must be at least one')
    it = iter(iterable)
    while (batch := tuple(islice(it, n))):
        yield batch

def chunked_tokens(text, encoding_name, chunk_length):
    encoding = tiktoken.get_encoding(encoding_name)
    tokens = encoding.encode(text)
    chunks_iterator = batched(tokens, chunk_length)
    yield from chunks_iterator

def len_safe_get_embedding(text, model=EMBEDDING_MODEL, max_tokens=EMBEDDING_CTX_LENGTH, 
                           encoding_name=EMBEDDING_ENCODING, average=True):
    #if average is true it will return the average embedding.
    #else jsut returns the list of embeddings for chunks

    chunk_embeddings = []
    chunk_lens = []
    for chunk in chunked_tokens(text, encoding_name=encoding_name, chunk_length=max_tokens):
        chunk_embeddings.append(get_embedding(chunk, model=model))
        chunk_lens.append(len(chunk))

    if average:
        chunk_embeddings = np.average(chunk_embeddings, axis=0, weights=chunk_lens)
        chunk_embeddings = chunk_embeddings / np.linalg.norm(chunk_embeddings)  # normalizes length to 1
        chunk_embeddings = chunk_embeddings.tolist() 
        
    chunk_embeddings = np.array(chunk_embeddings)
    
    #always return a 2d list of size (n, 1536) for n embeddings
    #there is probably a more efficient way to do this but it works for now
    
    if average:
        chunk_embeddings = chunk_embeddings.reshape(1, 1536)
        
    return chunk_embeddings 




# us_const = txt_from_file('constitution.txt')
# decl_ind = txt_from_file('declaration_of_independence.txt')

# average_embedding_vector = len_safe_get_embedding(us_const, average=True)
# chunks_embedding_vectors = len_safe_get_embedding(us_const, average=False)

# print(f"Setting average=True gives us a single {len(average_embedding_vector)}-dimensional embedding vector for our long text.")
# print(f"Setting average=False gives us {len(chunks_embedding_vectors)} embedding vectors, one for each of the chunks.")

# print(cosine_similarity(len_safe_get_embedding(us_const), get_embedding(decl_ind)))

'''
Next:
    
    1. Load in record pdfs and get embeddings. You probably want to take the average of the record.
    Although, not necessarily, you could just go chunk by chunk and get the similarity.
    2. Load in court cases to search in a vector database. Chunks, citation, text
'''