import streamlit as st
import pickle
#spacy
import spacy
from spacy.pipeline import EntityRuler
from spacy.lang.en import English
from spacy.tokens import Doc
#gensim
import gensim
from gensim import corpora
#Visualization
from spacy import displacy
import pyLDAvis
import pyLDAvis.gensim_models
#from wordcloud import WordCloud
import plotly.express as px
#import matplotlib.pyplot as plt
#Data loading/ Data manipulation
import pandas as pd
#import numpy as np
import jsonlines
#nltk
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
nltk.download(['stopwords','wordnet'])
nltk.download("en_core_web_lg")
#warning
import warnings
warnings.filterwarnings('ignore')
import locale
locale.getpreferredencoding = lambda: "UTF-8"
import resume

import spacy.cli

# # Download spaCy model if not already downloaded
# try:
#     nlp = spacy.load("en_core_web_lg")
# except OSError:
#     spacy.cli.download("en_core_web_lg")
    #nlp = spacy.load("en_core_web_lg")

#nlp = spacy.load("en_core_web_lg")

ruler = nltk.add_pipe("entity_ruler")
skill_pattern_path = r"C:\Users\johan\OneDrive\Desktop\DS Python\StreamLit\Employee_Moniter\jz_skill_patterns.jsonl"
ruler.from_disk(skill_pattern_path)
nltk.pipe_names

def phone_numbers(input_resume):
    # Regular expression pattern for matching mobile numbers
    mobile_number_pattern = r"\b\d{10}\b"
    # Preprocess the text using spaCy
    doc = nlp(input_resume)
    # Extract phone numbers using regular expressions
    phone_numbers = re.findall(mobile_number_pattern, input_resume)
    # Iterate over the spaCy document to find matches
    for entity in doc.ents:
        if entity.label_ == "PHONE_NUMBER":
            phone_numbers.append(entity.text)

    # Print the extracted phone numbers
    return phone_numbers
    
    
def parse_resume(input_resume):
    # Process the input resume using the spaCy model
    doc = nlp(input_resume)
    # Extract the contact information
    contact_info = {
        "email": None,
        "phone": phone_numbers(input_resume),
        "address": None
    }
    for token in doc:
        if token.like_email:
            contact_info["email"] = token.text
    # Extract the skills
    skills = []
    skills_section = False
    for token in doc:
        if token.text == "SKILLS":
            skills_section = True
            continue
        if skills_section:
            if token.is_alpha:
                skills.append(token.text)

    return contact_info, skills

def extract_email(email):
    email = re.findall("([^@|\s]+@[^@]+\.[^@|\s]+)", email)
    if email:
        try:
            return email[0].split()[0].strip(';')
        except IndexError:
            return None
        
def get_skills(text):
    doc = nlp(text)
    myset = []
    subset = []
    for ent in doc.ents:
        if ent.label_ == "SKILL":
            subset.append(ent.text)
    myset.append(subset)
    return subset

def unique_skills(x):
    return list(set(x))

# input_skills = "Data Science, Data Analysis, Database, SQL, Machine Learning, tableau, data mining, pandas, numpy, nlp, python, mongodb"


