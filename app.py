import streamlit as st
import pickle
import pandas as pd
import numpy as np
import json
import requests

import string
import nltk
nltk.download('wordnet')
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Functions for Bag of Words
def remove_punctutations(text):
    for punc in string.punctuation:
        text = text.replace(punc, '')
    return text.lower()



def root_word(text, method = 'lemma'):
    words = text.split()    
    if method == 'lemma':
        lemma = WordNetLemmatizer()
        root_words  = [lemma.lemmatize(word) for word in words] 
    elif method == 'stem':
        stem = PorterStemmer()
        root_words  = [stem.stem(word) for word in words] 
    else:
        raise SyntaxError("Method should be either 'lemma' or 'stem'")    
    return ' '.join(root_words)


stopwords = set("my a has with its of as on her his or our their is in by up says say for at to and the s he may -".split())
def stopword_removal(text, stopwords = stopwords):
    s_list = [word for word in text.split() if word not in stopwords]
    str_ = ' '.join(s_list)   
    return str_



def query_run(query, embedding, vectorizer, root_method = 'stem', stop_word_removal = True):    
    query_emb_array = vectorizer.transform([query]).toarray()
    if stop_word_removal == True:
        query = stopword_removal(remove_punctutations(query))
    query = root_word(query, method = root_method)
    cosine_sim = cosine_similarity(query_emb_array, embedding)    
    return cosine_sim



def find_top_n_results(df, query, embedding, vectorizer, root_method = 'stem', n=5, stop_word_removal = True ):
    cosine_sim_values = query_run(query, embedding, vectorizer, root_method, stop_word_removal)
    x = np.argsort(cosine_sim_values, axis = 1)[0,-n:]
    similarity_df = pd.DataFrame(columns=['Result','Similarity'])
    if type(df) != 'pandas.core.frame.DataFrame':
        new_df = pd.DataFrame(df)
    for i in x:
        similarity_df = similarity_df.append({'Result': new_df.loc[i,0], 
                                              'Similarity' : cosine_sim_values[0,i]},
                                              ignore_index = True)
    avg_sim_top_n = np.mean(cosine_sim_values[0,x])
    return similarity_df 


model_W2V = pickle.load(open('./models/w2v_similarity.pkl', 'rb'))

# Function for Word2Vec and GLoVE
def similarity_w2vglove(query, n_return, cluster_dict, clusters_model):
    query_emb_array = []

    q_nlp = nlp(query.lower())    
    for token in q_nlp:
        tok_lem = str(remove_punctutations(token.lemma_))
        if (tok_lem not in stopwords):
            if tok_lem in model_W2V.key_to_index:
                query_emb_array.append(model_W2V[tok_lem])
            else:
                try:
                    query_emb_array.append(model_W2V[tok_lem.capitalize()])
                except:
                    query_emb_array.append(np.zeros(model_W2V.vector_size))
    
    query_emb_array = np.array(query_emb_array).mean(axis=0).reshape(1,300)


    cluster_centroids = clusters_model.cluster_centers_
    similarity = cosine_similarity(query_emb_array, cluster_centroids)


    cluster_number = np.array(similarity).argmax()
    cluster_df = cluster_dict['Cluster %s'%cluster_number]

    cluster_embedding = cluster_df['vector'].values
    
    similarity_to_title = [] 

    for i in range (0, cluster_df.shape[0]):
        similarity_to_title.append(cosine_similarity(query_emb_array,cluster_embedding[i].reshape(1,300)))

    for i in range (0,len(similarity_to_title)):
        similarity_to_title[i] = similarity_to_title[i][0][0]

    cluster_df['similarity'] = similarity_to_title

    cluster_df = cluster_df.sort_values(by='similarity', ascending = False).reset_index()

    result = pd.DataFrame({
                            'Result' : cluster_df['Titles'].iloc[0:n_return],
                            'Similarity' : cluster_df['similarity'].iloc[0:n_return]
                            }, columns=['Result','Similarity'])
    
    return result

file = open('./data_world_example.json', 'r')
data = json.loads(file.read())
file.close()
titles =[]

for i in range(0,len(data)):
    titles.append(data[i]['title'])



def main():
    st.title('Headline Finder')
    
    text, n_results = st.columns([7,2])

    with text:
        text = st.text_input("Please input your search query")

    with n_results:
        n_results = st.number_input('Number of top results', step=1, min_value=1, max_value=10)


    bow_c, w2v_c, glv_c, w2vc_c, svd_c = st.columns(5)
    with bow_c:
        bow_c = st.checkbox('Bag of Words')
    with w2v_c:
        w2v_c = st.checkbox('Word2Vec (Pre-Trained)')
    with glv_c:
        glv_c = st.checkbox('GLoVe (Pre-Trained)')
    with w2vc_c:
        w2vc_c = st.checkbox('Word2Vec (Customized)')
    with svd_c:
        svd_c = st.checkbox('LSA / SVD')

    if bow_c == True:
        vectorizer = CountVectorizer(binary=True)
        embedding = vectorizer.fit(titles).transform(titles).toarray()
        bow_result = find_top_n_results(titles, text, embedding, vectorizer, root_method = 'stem', n=n_results, stop_word_removal=True)
        bow_result = bow_result.sort_values('Similarity', ascending=False, ignore_index=True)
        
    if w2v_c == True:
        w2v_clusters = pickle.load(open('./models/w2v_clusters.pkl', 'rb'))
        w2v_results = similarity_w2vglove(text, n_results, w2v_clusters)

    models = dict()

    model_list = [["Bag of Words" , bow_c], ["Word2Vec (Pre-Trained)" , w2v_c], ["GLoVe (Pre-Trained)",  glv_c], ["Word2Vec (Customized)" , w2vc_c], ["LSA / SVD", svd_c]]
    
    for i,j in model_list:
        models[i] = j


    columns = []
    for i in models.keys():
        if models[i] == True:
            columns.append(i)

    header = pd.MultiIndex.from_product([columns,
                                     ['Headline','Similarity']],
                                    names=['model','similarity'])

    result = pd.DataFrame(columns=header)

    if bow_c == True:
        for i in range(0,len(bow_result)):
            result.loc[i,('Bag of Words','Headline')] = bow_result['Result'][i]
            result.loc[i,('Bag of Words','Similarity')] = bow_result['Similarity'][i]

            
    if w2v_c == True:
        for i in range(0,len(w2v_results)):
            result.loc[i,('Word2Vec (Pre-Trained)','Headline')] = w2v_results['Result'][i]
            result.loc[i,('Word2Vec (Pre-Trained)','Similarity')] = w2v_results['Similarity'][i]

    # for i,j in model_list:
    #     st.write('Status of ',i,': ',j)

    if result.shape == (0,0):
        st.write('Please select an option above.')
    else:
        st.write(result)    

main()