import streamlit as st
import streamlit_authenticator as stauth
import datetime
import re
import sqlite3

# login = st.button('Login')
# signup = st.button('SignIn')

def insert_user(email, username, password):
    with sqlite3.connect('https://github.com/JOHANDILEEP21/Resume_parsing/blob/d22045305aa1d8eab6ed082916126a796fe00970/user_db.db') as conn:
        cursor = conn.cursor()
        try:
            conn.execute("BEGIN")
            # Perform multiple operations
            cursor.execute('INSERT INTO users (email, username, password) VALUES (?, ?, ?)', (email, username, password))
            conn.commit()
            #conn.close()
            st.write('SignUp Successfull')
        except Exception as e:
            conn.rollback()
            st.warning(f"Error: {str(e)}")

            
def fetch_users():
    conn = sqlite3.connect('https://github.com/JOHANDILEEP21/Resume_parsing/blob/d22045305aa1d8eab6ed082916126a796fe00970/user_db.db')
    cursor = conn.cursor()
    g = cursor.execute('select * from users')
    users = g.fetchall()
    return users
    
    
def get_user_emails():
    conn = sqlite3.connect('https://github.com/JOHANDILEEP21/Resume_parsing/blob/d22045305aa1d8eab6ed082916126a796fe00970/user_db.db')
    cursor = conn.cursor()
    g = cursor.execute('select * from users')
    data = g.fetchall()
    emails = []
    for user in data:
        emails.append(user[1])
    return emails

def get_usernames():
    conn = sqlite3.connect('https://github.com/JOHANDILEEP21/Resume_parsing/blob/d22045305aa1d8eab6ed082916126a796fe00970/user_db.db')
    cursor = conn.cursor()
    g = cursor.execute('select * from users')
    data = g.fetchall()
    usernames = []
    for user in data:
        usernames.append(user[2])
    return usernames
    
def validate_email(email):
    pattern = "^[a-zA-Z0-9]+@[a-zA-Z0-9]+\.[a-z]{1,3}$"
    if re.match(pattern, email):
        return True
    return False

def validate_username(username):
    pattern = "^[a-zA-Z0-9]+$"
    if re.match(pattern, username):
        return True
    return False
    
def sign_up():
    conn = sqlite3.connect('https://github.com/JOHANDILEEP21/Resume_parsing/blob/d22045305aa1d8eab6ed082916126a796fe00970/user_db.db')
    cursor = conn.cursor()
    create_table_sql = '''
    CREATE TABLE IF NOT EXISTS users (
        id INTEGER PRIMARY KEY,
        email text,
        username text,
        password text
    );'''
    cursor.execute(create_table_sql)
    conn.commit()
    with st.form(key='signup',clear_on_submit=True):
        st.subheader(':green[Signup]')
        email = st.text_input(':blue[Email]', placeholder='Enter Your Email')
        username = st.text_input(':blue[Username]', placeholder='Enter Your Username')
        password1 = st.text_input(':blue[Password]', placeholder='Enter Your Username', type='password')
        password2 = st.text_input(':blue[Confirm Password]', placeholder='Confirm Your Password', type='password')
        
        if email:
            if validate_email(email):
                if email not in get_user_emails():
                    if validate_username(username):
                        if username not in get_usernames():
                            if len(username) >= 2:
                                if len(password1) >=6:
                                    if password1 == password2:
                                        # add user to DB
                                        hashed_password = stauth.Hasher([password2]).generate()
                                        insert_user(email, username, hashed_password[0])
                                        st.success('Account Created Successfull')
                                        st.balloons()
                                    else:
                                        st.warning('Passwords Do not Match')
                                else:
                                    st.warning('Password is too Short')
                            else:
                                st.warning('Username is too short')
                        else:
                            st.warning('Username Already Exists')
                    else:
                        st.warning('Invalid Username')
                else:
                    st.warning('Email Already Exists')

            else:
                st.warning('Invalid Email')
                
        btn1, btn2, btn3, btn4, btn5 = st.columns(5)
        
        with btn3:
            st.form_submit_button('Sign Up')



