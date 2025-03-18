import streamlit as st
import pandas as pd
import plotly.express as px

from data import load_data, preprocess_data
from recommender import get_recommendations, get_recommendations_by_preferences

# Configure page
st.set_page_config(page_title="Fragrance Recommender", layout="wide")
st.title("AI Fragrance Recommendation System")
st.markdown("Find your perfect scent match based on notes, accords, and preferences.")

# Load and preprocess data
df = load_data()
if df is not None:
    df, features, notes_classes, accords_classes = preprocess_data(df)
    
    # Sidebar for selecting recommendation method
    st.sidebar.title("Find Your Perfect Fragrance")
    rec_method = st.sidebar.radio(
        "How would you like to discover fragrances?",
        ["By Similar Fragrance", "By Preferences"]
    )
    
    if rec_method == "By Similar Fragrance":
        search_term = st.sidebar.text_input("Search for a fragrance by name:")
        if search_term:
            filtered_df = df[df['Perfume'].str.contains(search_term, case=False, na=False, regex=False)]
            if not filtered_df.empty:
                st.sidebar.subheader("Select a fragrance:")
                options = {i: f"{row['Perfume']} by {row['Brand']}" for i, row in filtered_df.iterrows()}
                selected_index = st.sidebar.selectbox(
                    "Choose a fragrance:",
                    list(options.keys()),
                    format_func=lambda x: options[x]
                )
                
                # Display selected fragrance details
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
                    st.write("**Top Notes:** " + str(df.loc[selected_index, 'Top']))
                    st.write("**Middle Notes:** " + str(df.loc[selected_index, 'Middle']))
                    st.write("**Base Notes:** " + str(df.loc[selected_index, 'Base']))
                    accords = [df.loc[selected_index, col] for col in df.columns if 'Main Accord' in col and df.loc[selected_index, col]]
                    st.write("**Main Accords:** " + ", ".join([str(a) for a in accords if str(a) != 'nan']))
                
                # Get recommendations based on the selected fragrance
                num_recs = st.slider("Number of recommendations:", 5, 20, 10)
                recommendations = get_recommendations(df, features, selected_index, num_recs)
                
                st.header("Recommended Fragrances")
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
        all_notes = sorted(list(set(notes_classes)))
        all_accords = sorted(list(set(accords_classes)))
        
        gender_pref = st.sidebar.radio("Preferred Gender Category:", ["All", "Women", "Men", "Unisex"])
        selected_notes = st.sidebar.multiselect(
            "Select notes you enjoy:",
            all_notes,
            help="Select one or more fragrance notes you prefer"
        )
        selected_accords = st.sidebar.multiselect(
            "Select accords/scent families you enjoy:",
            all_accords,
            help="Select one or more fragrance families you prefer"
        )
        year_range = st.sidebar.slider(
            "Launch Year Range:",
            int(df['Year'].min()),
            int(df['Year'].max()),
            (int(df['Year'].min()), int(df['Year'].max()))
        )
        if st.sidebar.button("Find My Fragrances"):
            if not selected_notes and not selected_accords:
                st.sidebar.warning("Please select at least one note or accord.")
            else:
                recommendations = get_recommendations_by_preferences(
                    df, features, selected_notes, selected_accords, gender_pref, 
                    notes_classes, accords_classes, top_n=15
                )
                # Filter recommendations by year range
                recommendations = recommendations[
                    (recommendations['Year'] >= year_range[0]) & 
                    (recommendations['Year'] <= year_range[1])
                ].reset_index(drop=True)
                
                if len(recommendations) > 0:
                    st.header("Recommended Fragrances Based on Your Preferences")
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
    
    # Data insights section with interactive visualizations
    st.header("Explore the Fragrance World")
    tabs = st.tabs(["Popular Notes", "Brands", "Ratings", "Years"])
    
    with tabs[0]:
        top_notes = [note.strip() for notes in df['Top'].dropna() 
                     for note in str(notes).split(',') if note.strip()]
        middle_notes = [note.strip() for notes in df['Middle'].dropna() 
                        for note in str(notes).split(',') if note.strip()]
        base_notes = [note.strip() for notes in df['Base'].dropna() 
                      for note in str(notes).split(',') if note.strip()]
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
        fig = px.histogram(
            df[df['Rating Value'] > 0],
            x='Rating Value',
            nbins=20,
            title='Distribution of Fragrance Ratings'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        brand_ratings = df.groupby('Brand')['Rating Value'].agg(['mean', 'count']).reset_index()
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
        yearly_counts = df['Year'].value_counts().sort_index()
        fig = px.line(
            x=yearly_counts.index,
            y=yearly_counts.values,
            markers=True,
            labels={'x': 'Year', 'y': 'Number of Fragrances'},
            title='Fragrance Releases by Year'
        )
        st.plotly_chart(fig, use_container_width=True)
        
        df['Decade'] = (df['Year'] // 10) * 10
        decades = sorted(df['Decade'].dropna().unique())
        
        if st.checkbox("Show Top Notes by Decade"):
            selected_decade = st.select_slider("Choose a decade:", options=decades)
            decade_df = df[df['Decade'] == selected_decade]
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
