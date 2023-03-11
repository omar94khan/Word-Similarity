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



def stopword_removal(text, stopwords = set("my a has with its of as on her his or our their is in by up says say for at to and the s he may -".split())):
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


file = open('./data_world_example.json', 'r')
data = json.loads(file.read())
file.close()
titles =[]

for i in range(0,len(data)):
    titles.append(data[i]['title'])



def main():

    st.write(titles)
    st.title('Headline Finder')
    
    text, n_words = st.columns([7,2])

    with text:
        text = st.text_input("Please input your search query")

    with n_words:
        n_words = st.number_input('Number of top results', step=1, min_value=1, max_value=10)


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


    st.write('Status of text: ', text)
    st.write('Status of n_words: ', n_words)

    if bow_c == True:
        vectorizer = CountVectorizer(binary=True)
        embedding = vectorizer.fit(titles).transform(titles).toarray()
        result = find_top_n_results(titles, text, embedding, vectorizer, root_method = 'stem', n=n_words, stop_word_removal=True).sort_values('Similarity', ascending=False, ignore_index=True)
        st.write(result)

    models = dict()

    model_list = [["Bag of Words" , bow_c], ["Word2Vec (Pre-Trained)" , w2v_c], ["GLoVe (Pre-Trained)",  glv_c], ["Word2Vec (Customized)" , w2vc_c], ["LSA / SVD", svd_c]]
    
    for i,j in model_list:
        models[i] = j


    columns = []
    for i in models.keys():
        if models[i] == True:
            columns.append(i)

    result = pd.DataFrame(columns=columns)

    for i,j in model_list:
        st.write('Status of ',i,': ',j)

    st.write(result)

    # if file is not None:
    #     df = pd.read_csv(file)
    #     st.write("The dataset you uploaded is:")
    #     st.write(df)
    #     st.write("The dataset is of shape: ", df.shape)

    #     df2 = pd.DataFrame()
        
    #     columns = ['Time','V1','V2','V3','V4','V5','V6','V7','V8','V9','V10','V11','V12',
    #                 'V13','V14','V15','V16','V17','V18','V19','V20','V21','V22','V23',
    #                 'V24','V25','V26','V27','V28','Amount']

    #     for col in columns:
    #         df2[col] = df[col]

    #     df2 = pd.DataFrame(MinMaxScaler().fit(df2).transform(df2), columns=df2.columns)
        
    #     classifier = pickle.load(open('RandomForrestClassifier_df4x.pkl', 'rb'))
    #     result_df = pd.DataFrame(classifier.predict_proba(df2))[1]
            
    #     st.write("Output DataFrame depicting probability of the transaction being fraudulant.")
    #     st.write(result_df)
    #     st.write("Result shape: ", result_df.shape)

    #     st.download_button(label="Download the results",
    #                         data = result_df.to_csv().encode('utf-8'),
    #                         file_name = "fraud_probability.csv",
    #                         mime = 'text/csv')

main()