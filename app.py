import streamlit as st
import pandas as pd
import numpy as np
import pickle

def main():
    st.title('Word Similarity')
    
    text = st.text_input("Please input the word for which you would like to find the similarity")
    method = st.radio('Mathod of word comparison', label=['Bag of Words'])#,'Word2Vec (pre-trained)', 'GLoVe (pre-trained)','Word2Vec (customized)','LSA/SVD'])

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