import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title('Word Similarity')
    
    text, n_words = st.columns([6,2])

    with text:
        st.text_input("Please input the word for which you would like to find the similarity")

    with n_words:
        st.number_input('Number of similar words', step=1)

    # text = st.text_input("Please input the word for which you would like to find the similarity")
    # n_words = st.number_input('Please specify the number of similar words you want', step=1)

    bow = st.checkbox('Bag of Words')
    w2v = st.checkbox('Word2Vec (Pre-Trained)')
    glv = st.checkbox('GLoVe (Pre-Trained)')
    w2vc = st.checkbox('Word2Vec (Customized)')
    svd = st.checkbox('LSA / SVD')
    
    all_models = [bow,w2v,glv,w2vc,svd]

    result = pd.DataFrame()

    for i in all_models:
        if i:
            result.append(pd.DataFrame(columns=str(i)), ignore_index = True)

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