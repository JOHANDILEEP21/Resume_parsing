import re
import nltk
import PyPDF2
import pandas as pd
import os, models
import streamlit as st
from pathlib import Path
import pandas as pd
import plotly.express as px
import models
from resume_screen import get_skills
import resume_screen

def resume_parsing():
    try:
        pos = []
        position = st.text_input("Applying Position")
        pos.append(position)
        #st.text(pos)

        mdl = models.main()
        # st.write(mdl)

        resume_text, mdd = mdl
        # st.text(resume_text)
        if mdd:
            input_resume = resume_text
            input_resume = ''.join(input_resume)
            
            contact_info, resume_skills = resume_screen.parse_resume(input_resume)
            
            required_skills = st.text_input('Enter the skills that you need')
            #input_skills = "Data Science, Data Analysis, Database, SQL, Machine Learning, tableau, data mining, pandas, numpy, nlp, python, mongodb"

            req_skills = required_skills.lower().split(", ")
            #st.text(['Required Skills: ', req_skills])
            # resume_skills = resume_screen.get_skills(input_resume.lower())  
            resume_skills = resume_screen.unique_skills(get_skills(get_skills.lower()))
            st.text(resume_skills)
            resume_skills = set(resume_skills)
            #st.text(resume_skills)
            score = 0
            for x in req_skills:
                if x in resume_skills:
                    score += 1
            req_skills_len = len(req_skills)
            st.text([req_skills_len, score])
            match = round(score / req_skills_len * 100, 1)
            #st.text(match)
            st.text(f"The current Resume is {match}% matched to your requirements")

        else:
            st.warning('Please Check your code')
    
    except Exception as e:
        st.warning(f'Please feed the above credentials {e}')

#resume_parsing()



