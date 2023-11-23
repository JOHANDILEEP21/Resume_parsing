import streamlit as st
import streamlit_authenticator as stauth
from signup import fetch_users
#from linear_regression import linear_models
import login_page
from login_page import model

def main_page():
    try:
        users = fetch_users()
        emails = []
        usernames = []
        passwords = []

        for user in users:
            emails.append(user[1])
            usernames.append(user[2])
            passwords.append(user[3])

        credentials = {'usernames':{}}

        for index in range(len(emails)):
            credentials['usernames'][usernames[index]] = {'name':emails[index], 'password':passwords[index]}
        #st.write(credentials)

        Authenticator = stauth.Authenticate(credentials, cookie_name='Streamlit',key='abcdef', cookie_expiry_days=4)
        # st.write(Authenticator)

        email, authentication_status, username, = Authenticator.login(':green[Login]', 'main')
        #st.text(authentication_status)
        info, info1 = st.columns(2)
        
        if username:
            if username in usernames:
                if authentication_status:
                    return credentials, Authenticator, username, usernames
                    
                elif not authentication_status:
                    with info:
                        st.error('Incorrect password or username')
                else:
                    with info:
                        st.warning('Please feed in your credentials')
            else:
                with info:
                    st.warning('Username does not exist, Please sign up')
        else:
            with info:
                st.warning('Please feed in your credentials')

    except Exception as e:
        st.success(e)
