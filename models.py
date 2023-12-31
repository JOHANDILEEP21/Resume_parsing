import streamlit as st
import pickle
import re
import pdfplumber
import requests
from io import BytesIO
import zipfile

def remove_null_characters(text):
    return text.replace('\x00', '')

def clean_resume(resume_text):
    clean_text = re.sub('http\S+\s*', ' ', resume_text)
    clean_text = re.sub('RT|cc', ' ', clean_text)
    clean_text = re.sub('#\S+', '', clean_text)
    #clean_text = re.sub('@\S+', '  ', clean_text)
    clean_text = re.sub('[%s]' % re.escape("""!"#$%&'()*+,-:;<=>?[]^_`{|}~"""), ' ', clean_text)
    clean_text = re.sub(r'[^\x00-\x7f]', r' ', clean_text)
    clean_text = re.sub('\s+', ' ', clean_text)
    clean_text = re.sub(r'(\\n|\n)', ' ', clean_text)
    
    return clean_text

def extract_data(feed):
    data = []
    with pdfplumber.open(feed) as pdf:
        pages = pdf.pages
        #st.write(pages)
        for p in pages:
            tables = p.extract_tables()
            data.append(tables)
    #st.write(data)
    return data

# web app
def main():
    try:
        #st.title("Resume Screening App")
        uploaded_file = st.file_uploader('Upload Resume', type=['txt','pdf'])
        #st.text(type(uploaded_file))
        #st.write(md)
        if uploaded_file is not None:
            try:
                df = extract_data(uploaded_file)
                resume_text = str(df)
                #st.text(df)
                #resume_bytes = uploaded_file.read()
                #resume_text = remove_null_characters(uploaded_file.read().decode('utf-8'))
            except UnicodeDecodeError as uce:
                # If UTF-8 decoding fails, try decoding with 'latin-1'
                #resume_text = remove_null_characters(uploaded_file.read().decode('latin-1'))
                st.write(uce)

            zip_file_path = 'clf.zip'
            pickle_file_name = 'clf.pkl'
    
            with zipfile.ZipFile(zip_file_path, 'r') as zip_file:
                with zip_file.open(pickle_file_name) as pickle_file:
                    # Load the pickled data
                    clf = pickle.load(pickle_file)
            tfidf = pickle.load(open('tfidf.pkl', 'rb'))
            #clf, tfidf = md
            #st.write(clf, tfidf)
            #tfidf = pickle.load(open('https://github.com/JOHANDILEEP21/Resume_parsing/blob/main/tfidf.pkl', 'rb'))
            #tfidf = pickle.load(open('https://github.com/JOHANDILEEP21/Resume_parsing/blob/main/tfidf.pkl'))
            
            cleaned_resume = clean_resume(resume_text)
            cleaned_resume = str(cleaned_resume)
            #st.text(type(cleaned_resume))
            input_features = tfidf.transform([cleaned_resume])
            prediction_id = clf.predict(input_features)[0]
            #st.write(prediction_id)
    
            # Map category ID to category name
            category_mapping = {
                15: "Java Developer",
                23: "Testing",
                8: "DevOps Engineer",
                20: "Python Developer",
                24: "Web Designing",
                12: "HR",
                13: "Hadoop",
                3: "Blockchain",
                10: "ETL Developer",
                18: "Operations Manager",
                6: "Data Science",
                22: "Sales",
                16: "Mechanical Engineer",
                1: "Arts",
                7: "Database",
                11: "Electrical Engineering",
                14: "Health and fitness",
                19: "PMO",
                4: "Business Analyst",
                9: "DotNet Developer",
                2: "Automation Testing",
                17: "Network Security Engineer",
                21: "SAP Developer",
                5: "Civil Engineer",
                0: "Advocate",
            }
    
            category_name = category_mapping.get(prediction_id, "Unknown")
    
            st.write("Predicted Category:", category_name)
            
            return cleaned_resume, True
        # else:
        #     st.text('Error in this function')

    except Exception as e:
        st.warning(e)

