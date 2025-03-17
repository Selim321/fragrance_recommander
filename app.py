import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MultiLabelBinarizer
import plotly.express as px

# Set page config
st.set_page_config(page_title="Fragrance Recommender", layout="wide")

# App title and description
st.title("AI Fragrance Recommendation System")
st.markdown("Find your perfect scent match based on notes, accords, and preferences.")

# Load data
@st.cache_data
def load_data():
    try:
        # Try with semicolon as separator and different encodings
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

# Function to preprocess data
@st.cache_data
def preprocess_data(df):
    # Handle missing values
    df = df.fillna('')

    # Convert comma-separated decimals to periods
    df['Rating Value'] = df['Rating Value'].str.replace(',', '.').astype(float)
    
    # Convert 'Year' column to integer
    df['Year'] = pd.to_numeric(df['Year'], errors='coerce').astype('Int64')
    
    # Combine notes into single lists for processing
    df['all_notes'] = df.apply(
        lambda row: [note.strip() for note in 
                    (str(row['Top']) + ', ' + 
                    str(row['Middle']) + ', ' + 
                    str(row['Base'])).split(',') 
                    if note.strip()], axis=1
    )

    # Combine accords
    accord_columns = [col for col in df.columns if 'mainaccord' in col]
    df['all_accords'] = df.apply(
        lambda row: [row[col] for col in accord_columns if row[col] and str(row[col]) != 'nan'], axis=1
    )
    
    # Create feature vectors
    mlb_notes = MultiLabelBinarizer()
    notes_matrix = mlb_notes.fit_transform(df['all_notes'])
    
    mlb_accords = MultiLabelBinarizer()
    accords_matrix = mlb_accords.fit_transform(df['all_accords'])
    
    # Create gender feature (one-hot encoded)
    gender_map = {'men': [1, 0, 0], 'women': [0, 1, 0], 'unisex': [0, 0, 1]}
    gender_features = np.array([gender_map.get(str(g).lower().strip(), [0, 0, 0]) for g in df['Gender']])
    
    # Combine features with weights
    # Notes (0.6), Accords (0.3), Gender (0.1)
    notes_weighted = notes_matrix * 0.6
    accords_weighted = accords_matrix * 0.3
    gender_weighted = gender_features * 0.1
    
    # Ensure all matrices have samples in rows
    features = np.hstack((
        notes_weighted, 
        accords_weighted,
        np.zeros((len(df), max(0, notes_weighted.shape[1] + accords_weighted.shape[1] - gender_weighted.shape[1]))),
        gender_weighted
    ))
    
    return df, features, mlb_notes.classes_, mlb_accords.classes_

# Function to get recommendations based on fragrance
def get_recommendations(df, features, index, top_n=10):
    # Calculate similarity
    sim_scores = cosine_similarity([features[index]], features)[0]
    
    # Get scores with indices
    scores_with_indices = list(enumerate(sim_scores))
    
    # Sort by similarity score
    scores_with_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Get top N most similar fragrances (excluding the input fragrance)
    top_indices = [i for i, _ in scores_with_indices[1:top_n+1]]
    
    # Return recommendations with similarity scores
    recommendations = df.iloc[top_indices].copy().reset_index(drop=True)
    recommendations['similarity'] = [scores_with_indices[i+1][1] for i in range(len(top_indices))]
    return recommendations

# Function to get recommendations based on notes and preferences
def get_recommendations_by_preferences(df, features, selected_notes, selected_accords, 
                                      gender_preference, notes_classes, accords_classes, top_n=10):
    # Create binary vectors
    user_notes_vector = np.zeros(len(notes_classes))
    for note in selected_notes:
        if note in notes_classes:
            idx = np.where(notes_classes == note)[0][0]
            user_notes_vector[idx] = 1
    
    user_accords_vector = np.zeros(len(accords_classes))
    for accord in selected_accords:
        if accord in accords_classes:
            idx = np.where(accords_classes == accord)[0][0]
            user_accords_vector[idx] = 1
    
    # Gender vector
    gender_map = {'men': [1, 0, 0], 'women': [0, 1, 0], 'unisex': [0, 0, 1], 'all': [0.33, 0.33, 0.33]}
    gender_vector = np.array(gender_map.get(gender_preference.lower(), [0.33, 0.33, 0.33]))
    
    # Apply weights
    notes_weighted = user_notes_vector * 0.6
    accords_weighted = user_accords_vector * 0.3
    gender_weighted = gender_vector * 0.1
    
    # Combine vectors
    user_vector = np.concatenate([
        notes_weighted,
        accords_weighted,
        np.zeros(max(0, features.shape[1] - len(notes_weighted) - len(accords_weighted) - len(gender_weighted))),
        gender_weighted
    ])
    
    # Ensure user vector is same length as features
    if len(user_vector) < features.shape[1]:
        user_vector = np.pad(user_vector, (0, features.shape[1] - len(user_vector)))
    elif len(user_vector) > features.shape[1]:
        user_vector = user_vector[:features.shape[1]]
    
    # Calculate similarity
    sim_scores = cosine_similarity([user_vector], features)[0]
    
    # Get recommendations
    indices = np.argsort(sim_scores)[::-1][:top_n]
    recommendations = df.iloc[indices].copy().reset_index(drop=True)
    recommendations['similarity'] = sim_scores[indices]
    return recommendations

# Load data
df = load_data()

if df is not None:
    # Preprocess data
    df, features, notes_classes, accords_classes = preprocess_data(df)
    
    # Sidebar
    st.sidebar.title("Find Your Perfect Fragrance")
    
    # Choose recommendation method
    rec_method = st.sidebar.radio(
        "How would you like to discover fragrances?",
        ["By Similar Fragrance", "By Preferences"]
    )
    
    if rec_method == "By Similar Fragrance":
        # Search by fragrance name
        search_term = st.sidebar.text_input("Search for a fragrance by name:")
        
        if search_term:
            filtered_df = df[df['Perfume'].str.contains(search_term, case=False, na=False)]
            
            if not filtered_df.empty:
                st.sidebar.subheader("Select a fragrance:")
                
                # Format options as "Name by Brand"
                options = {i: f"{row['Perfume']} by {row['Brand']}" 
                         for i, row in filtered_df.iterrows()}
                
                selected_index = st.sidebar.selectbox(
                    "Choose a fragrance:",
                    list(options.keys()),
                    format_func=lambda x: options[x]
                )
                
                # Show selected fragrance details
                st.header("Selected Fragrance")
                col1, col2 = st.columns([1, 2])
                
                with col1:
                    st.image("https://via.placeholder.com/150", caption=df.loc[selected_index, 'Perfume'])
                
                with col2:
                    st.subheader(f"{df.loc[selected_index, 'Perfume']}")
                    st.write(f"**Brand:** {df.loc[selected_index, 'Brand']}")
                    st.write(f"**Year:** {df.loc[selected_index, 'Year']}")
                    st.write(f"**Gender:** {df.loc[selected_index, 'Gender']}")
                    st.write(f"**Rating:** {df.loc[selected_index, 'Rating Value']} ({df.loc[selected_index, 'Rating Count']} votes)")
                    
                    # Display notes
                    st.write("**Top Notes:** " + str(df.loc[selected_index, 'Top']))
                    st.write("**Middle Notes:** " + str(df.loc[selected_index, 'Middle']))
                    st.write("**Base Notes:** " + str(df.loc[selected_index, 'Base']))
                    
                    # Display main accords
                    accords = [df.loc[selected_index, col] for col in df.columns if 'Main Accord' in col and df.loc[selected_index, col]]
                    st.write("**Main Accords:** " + ", ".join([str(a) for a in accords if str(a) != 'nan']))
                
                # Get and display recommendations
                num_recs = st.slider("Number of recommendations:", 5, 20, 10)
                recommendations = get_recommendations(df, features, selected_index, num_recs)
                
                st.header("Recommended Fragrances")
                
                # Display recommendations in cards
                num_cols = 3
                for i in range(0, len(recommendations), num_cols):
                    cols = st.columns(num_cols)
                    for j in range(num_cols):
                        if i + j < len(recommendations):
                            rec = recommendations.iloc[i+j]
                            with cols[j]:
                                st.image("https://via.placeholder.com/100")
                                st.subheader(rec['Perfume'])
                                st.write(f"**Brand:** {rec['Brand']}")
                                st.write(f"**Similarity:** {rec['similarity']:.2f}")
                                with st.expander("Details"):
                                    st.write(f"**Gender:** {rec['Gender']}")
                                    st.write(f"**Year:** {rec['Year']}")
                                    st.write(f"**Rating:** {rec['Rating Value']} ({rec['Rating Count']} votes)")
                                    st.write(f"**Top Notes:** {rec['Top']}")
            else:
                st.sidebar.warning("No fragrances found. Try another search term.")
    
    elif rec_method == "By Preferences":
        st.sidebar.subheader("Select Your Preferences")
        
        # Extract all unique notes and accords
        all_notes = sorted(list(set(notes_classes)))
        all_accords = sorted(list(set(accords_classes)))
        
        # Gender preference
        gender_pref = st.sidebar.radio("Preferred Gender Category:", 
                                      ["All", "Women", "Men", "Unisex"])
        
        # Note preferences with autocomplete
        selected_notes = st.sidebar.multiselect(
            "Select notes you enjoy:",
            all_notes,
            help="Select one or more fragrance notes you prefer"
        )
        
        # Accord preferences
        selected_accords = st.sidebar.multiselect(
            "Select accords/scent families you enjoy:",
            all_accords,
            help="Select one or more fragrance families you prefer"
        )
        
        # Launch year range
        year_range = st.sidebar.slider(
            "Launch Year Range:",
            int(df['Year'].min()),
            int(df['Year'].max()),
            (int(df['Year'].min()), int(df['Year'].max()))
        )
        
        # Get recommendations button
        if st.sidebar.button("Find My Fragrances"):
            if not selected_notes and not selected_accords:
                st.sidebar.warning("Please select at least one note or accord.")
            else:
                # Get recommendations
                recommendations = get_recommendations_by_preferences(
                    df, features, selected_notes, selected_accords, gender_pref, 
                    notes_classes, accords_classes, top_n=15
                )
                
                # Filter by year
                recommendations = recommendations[
                    (recommendations['Year'] >= year_range[0]) & 
                    (recommendations['Year'] <= year_range[1])
                ].reset_index(drop=True)
                
                # Display recommendations
                if len(recommendations) > 0:
                    st.header("Recommended Fragrances Based on Your Preferences")
                    
                    # Display in cards
                    num_cols = 3
                    for i in range(0, len(recommendations), num_cols):
                        cols = st.columns(num_cols)
                        for j in range(num_cols):
                            if i + j < len(recommendations):
                                rec = recommendations.iloc[i+j]
                                with cols[j]:
                                    st.image("https://via.placeholder.com/100")
                                    st.subheader(rec['Perfume'])
                                    st.write(f"**Brand:** {rec['Brand']}")
                                    st.write(f"**Match Score:** {rec['similarity']:.2f}")
                                    with st.expander("Details"):
                                        st.write(f"**Gender:** {rec['Gender']}")
                                        st.write(f"**Year:** {rec['Year']}")
                                        st.write(f"**Rating:** {rec['Rating Value']} ({rec['Rating Count']} votes)")
                                        st.write(f"**Top Notes:** {rec['Top']}")
                else:
                    st.warning("No fragrances match your criteria. Try adjusting your preferences.")
    
    # Data insights section
    st.header("Explore the Fragrance World")
    tabs = st.tabs(["Popular Notes", "Brands", "Ratings", "Years"])
    
    with tabs[0]:
        # Extract all notes
        top_notes = [note.strip() for notes in df['Top'].dropna() for note in str(notes).split(',') if note.strip()]
        middle_notes = [note.strip() for notes in df['Middle'].dropna() for note in str(notes).split(',') if note.strip()]
        base_notes = [note.strip() for notes in df['Base'].dropna() for note in str(notes).split(',') if note.strip()]
        
        all_notes_list = top_notes + middle_notes + base_notes
        note_counts = pd.Series(all_notes_list).value_counts().head(20)
        
        fig = px.bar(
            x=note_counts.index,
            y=note_counts.values,
            labels={'x': 'Note', 'y': 'Count'},
            title='Top 20 Most Popular Fragrance Notes'
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with tabs[1]:
        brand_counts = df['Brand'].value_counts().head(15)
        fig = px.pie(
            values=brand_counts.values,
            names=brand_counts.index,
            title='Top 15 Fragrance Brands by Number of Products'
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[2]:
        # Rating distribution
        fig = px.histogram(
            df[df['Rating Value'] > 0],
            x='Rating Value',
            nbins=20,
            title='Distribution of Fragrance Ratings'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Average rating by brand
        
        brand_ratings = df.groupby('Brand')['Rating Value'].agg(['mean', 'count']).reset_index()
        # Filter brands with at least 5 ratings
        brand_ratings = brand_ratings[brand_ratings['count'] >= 5].sort_values('mean', ascending=False).head(15)
        
        fig = px.bar(
            brand_ratings,
            x='Brand',
            y='mean',
            text_auto='.2f',
            color='count',
            color_continuous_scale='Blues',
            title='Top 15 Highest Rated Brands (with at least 5 fragrances)',
            labels={'mean': 'Average Rating', 'count': 'Number of Fragrances'}
        )
        st.plotly_chart(fig, use_container_width=True)

    with tabs[3]:
        # Fragrances by release year
        yearly_counts = df['Year'].value_counts().sort_index()
        
        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            markers=True,
            labels={'x': 'Year', 'y': 'Number of Fragrances'},
            title='Fragrance Releases by Year'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Popular notes by decade
        df['Decade'] = (df['Year'] // 10) * 10
        decades = sorted(df['Decade'].dropna().unique())
        
        if st.checkbox("Show Top Notes by Decade"):
            selected_decade = st.select_slider("Choose a decade:", options=decades)
            
            decade_df = df[df['Decade'] == selected_decade]
            
            # Extract notes from the selected decade
            decade_top_notes = [note.strip() for notes in decade_df['Top'].dropna() 
                            for note in str(notes).split(',') if note.strip()]
            
            if decade_top_notes:
                decade_note_counts = pd.Series(decade_top_notes).value_counts().head(10)
                
                fig = px.bar(
                    x=decade_note_counts.index,
                    y=decade_note_counts.values,
                    title=f'Top 10 Popular Notes in {selected_decade}s',
                    labels={'x': 'Note', 'y': 'Count'}
                )
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.write("No note data available for this decade.")

