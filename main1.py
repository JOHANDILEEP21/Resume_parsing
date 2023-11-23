import streamlit as st
import streamlit_authenticator as stauth
from streamlit_option_menu import option_menu
from signup import sign_up, fetch_users
#from linear_regression import linear_models
import login_page, main, resume, models
from PIL import Image
import pdfminer
from pdfminer.high_level import extract_pages

#st.set_page_config(page_title='UNIJACK')
# img = Image.open('https://github.com//JOHANDILEEP21//Resume_parsing//blob//6ffc07966a0b98f9534b3f128ddaae2ac6841c65//signup_page.jpg')
# st.sidebar.image(img)

class MultiApp:
    
    def __init__(self):
        self.apps = []
        
    def add_app(self, title, function):
        self.apps.append({
        'title':title,
        'function':function
        })
    def run():
        with st.sidebar:
            app = option_menu(
            menu_title = 'UniJack',
            options=['LogIn', 'SignUp'],
            menu_icon = 'chat-text-fill',
            default_index=1,
            styles={
            'container':{'padding':'5!important','background-color':'black'},
            'icon':{'color':'white', 'font-size':'23pz'},
            'nav-link':{'color':'white', 'font-size':'20px', 'text-align':'left', 'margin':'0px', '--hover-color':'blue'},
            'nav-link-selected':{'background-color':'#02ab21'},})
            
        if app == 'LogIn':
            mp = main.main_page()
            st.text(mp)
            if mp:
                credentials, Authenticator, username, usernames = mp
                login_page.login_page(credentials, Authenticator, username, usernames)
            
        if app == 'SignUp':
            sign_up()
            
    run()
            


