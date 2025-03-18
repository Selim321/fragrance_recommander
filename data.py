import pandas as pd
import numpy as np
import streamlit as st
from sklearn.preprocessing import MultiLabelBinarizer

@st.cache_data
def load_data():
    try:
        encodings = ['latin1', 'ISO-8859-1', 'cp1252', 'utf-8']
        for encoding in encodings:
            try:
                df = pd.read_csv('fra_cleaned.csv', sep=';', encoding=encoding)
                st.success(f"Successfully loaded data with {encoding} encoding and semicolon separator")
                return df
            except UnicodeDecodeError:
                continue
            except Exception as e:
                st.error(f"Error with {encoding}: {e}")
        st.error("Could not load the file. Please check the format.")
        return None
    except Exception as e:
        st.error(f"General error: {e}")
        return None

@st.cache_data
def preprocess_data(df):
    # Fill missing values and convert types
    df = df.fillna('')
    df['Rating Value'] = df['Rating Value'].str.replace(',', '.').astype(float)
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    
    # Combine notes (Top, Middle, Base) into a list for each row
    df['all_notes'] = df.apply(
        lambda row: [note.strip() for note in 
                     (str(row['Top']) + ', ' + str(row['Middle']) + ', ' + str(row['Base'])).split(',')
                     if note.strip()],
        axis=1
    )
    
    # Combine accords (based on columns containing 'mainaccord' regardless of case)
    accord_columns = [col for col in df.columns if 'mainaccord' in col.lower()]
    df['all_accords'] = df.apply(
        lambda row: [row[col] for col in accord_columns if row[col] and str(row[col]) != 'nan'],
        axis=1
    )
    
    # Create feature vectors for notes and accords using MultiLabelBinarizer
    mlb_notes = MultiLabelBinarizer()
    notes_matrix = mlb_notes.fit_transform(df['all_notes'])
    
    mlb_accords = MultiLabelBinarizer()
    accords_matrix = mlb_accords.fit_transform(df['all_accords'])
    
    # Create one-hot encoded gender features
    gender_map = {'men': [1, 0, 0], 'women': [0, 1, 0], 'unisex': [0, 0, 1]}
    gender_features = np.array([gender_map.get(str(g).lower().strip(), [0, 0, 0]) for g in df['Gender']])
    
    # Apply weights and combine features
    notes_weighted = notes_matrix * 0.6
    accords_weighted = accords_matrix * 0.3
    gender_weighted = gender_features * 0.1
    
    # Directly concatenate weighted features
    features = np.hstack((notes_weighted, accords_weighted, gender_weighted))
    
    return df, features, mlb_notes.classes_, mlb_accords.classes_
