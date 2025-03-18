import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

def get_recommendations(df, features, index, top_n=10):
    # Compute cosine similarity scores between the selected fragrance and all others
    sim_scores = cosine_similarity([features[index]], features)[0]
    scores_with_indices = list(enumerate(sim_scores))
    scores_with_indices.sort(key=lambda x: x[1], reverse=True)
    
    # Exclude the first entry (the selected fragrance itself) and select the next top_n
    top_indices = [i for i, _ in scores_with_indices[1:top_n+1]]
    
    recommendations = df.iloc[top_indices].copy().reset_index(drop=True)
    recommendations['similarity'] = [scores_with_indices[i+1][1] for i in range(len(top_indices))]
    return recommendations

def get_recommendations_by_preferences(df, features, selected_notes, selected_accords, 
                                       gender_preference, notes_classes, accords_classes, top_n=10):
    # Create binary vectors for selected notes and accords
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

    # Map gender preference to a vector
    gender_map = {'men': [1, 0, 0], 'women': [0, 1, 0], 'unisex': [0, 0, 1], 'all': [0.33, 0.33, 0.33]}
    gender_vector = np.array(gender_map.get(gender_preference.lower(), [0.33, 0.33, 0.33]))
    
    # Apply weights and combine into a single user vector
    notes_weighted = user_notes_vector * 0.6
    accords_weighted = user_accords_vector * 0.3
    gender_weighted = gender_vector * 0.1
    user_vector = np.concatenate([notes_weighted, accords_weighted, gender_weighted])
    
    # Calculate cosine similarity between the user vector and all feature vectors
    sim_scores = cosine_similarity([user_vector], features)[0]
    indices = np.argsort(sim_scores)[::-1][:top_n]
    
    recommendations = df.iloc[indices].copy().reset_index(drop=True)
    recommendations['similarity'] = sim_scores[indices]
    return recommendations
