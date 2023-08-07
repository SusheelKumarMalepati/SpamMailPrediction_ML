# -*- coding: utf-8 -*-
"""
Created on Mon Aug  7 13:36:45 2023

@author: HP
"""
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st

maildata=pd.read_csv('mail_data.csv')
x=maildata['Message']
y=maildata['Category']
x_train,x_test,y_train,y_test=train_test_split(x,y,random_state=3,test_size=0.2)
#convert text data to numerical data/feature vector by feature extraction
feature=TfidfVectorizer(min_df=1,stop_words='english',lowercase=True)
feature.fit_transform(x_train)

loaded_model=pickle.load(open('SpamMail_model.sav','rb'))

def spam(input):
    #input=str(input)
    feat=feature.transform(input)
    prediction=loaded_model.predict(feat)
    if prediction[0]==0:
        return 'spam mail'
    else:
        return 'ham mail'
        
def main():
    #title
    st.title("spam mail prediction")

    #input data
    text=st.text_input("enter mail content")
    
    #prediction
    output=''
    #button for prediction
    if st.button("check mail"):
        output=spam([text])
    st.success(output)

if __name__=='__main__':
    main()
