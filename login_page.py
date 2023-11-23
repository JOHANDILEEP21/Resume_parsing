import streamlit as st
import streamlit_authenticator as stauth
from signup import sign_up, fetch_users
import resume
import pandas as pd
import models
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.multiclass import OneVsRestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import pickle
import models

def model():
    try:
        df = pd.read_csv(r'https://github.com/JOHANDILEEP21/Resume_parsing/blob/439539052f5014a9ffcc0d33818862cfda0104eb/UpdatedResumeDataSet.csv')

        df['Resume'] = df['Resume'].apply(lambda x: models.clean_resume(x))

        le = LabelEncoder()
        le.fit(df['Category'])
        df['Category'] = le.transform(df['Category'])
        #st.dataframe(df)

        tfidf = TfidfVectorizer(stop_words='english')
        tfidf.fit(df['Resume'])
        requredTaxt  = tfidf.transform(df['Resume'])
        #st.write(requredTaxt)
        X_train, X_test, y_train, y_test = train_test_split(requredTaxt, df['Category'], test_size=0.2, random_state=42)

        clf = OneVsRestClassifier(KNeighborsClassifier())
        clf.fit(X_train,y_train)
        #ypred = clf.predict(X_test)
        #st.write(accuracy_score(y_test,ypred))

        pickle.dump(tfidf,open('tfidf.pkl','wb'))
        pickle.dump(clf, open('clf.pkl', 'wb'))
        
        # Load the trained classifier
        # clf = pickle.load(open('clf.pkl', 'rb'))
        st.write([tfidf, clf])

        return clf, tfidf
    
    except Exception as e:
        st.text(e)

def login_page(credentials, Authenticator, username, usernames):
    st.title(':red[UniJack]')
    st.header('Resume Screening App')
    try:
        info, info1 = st.columns(2)
        # let user see app
        st.sidebar.subheader(f'Welcome {username.capitalize()}')
        col1, col2, col3, col4, col5 = st.columns(5)
        # Now you can use the columns to add content
        with col5:
            Authenticator.logout('Log out') #, 'sidebar')
        md = model()
        start = st.checkbox('Click here to start')
        if start:
            res = resume.resume_parsing(md)
            res
        else:
            st.write('Please upload dataset for model training purposes')
                    
    except Exception as e:
        st.warning(e)
    
