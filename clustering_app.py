import os
import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from plotly import graph_objects as go
import plotly.express as px
import streamlit as st
import openai
from openai import OpenAI
from dotenv import load_dotenv

# Force reload of dotenv to ensure environment variables are loaded
load_dotenv(override=True)


# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="2025 IAP Student Assembly Voting Results Day 3", page_icon=":ballot_box:")


@st.cache_data
def data_load(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error loading Recommendations CSV file: {e}")


@st.cache_data
def load_suggestions():
    try:
        recs = pd.read_csv('results/all_suggestions_categorized.csv')
        return recs
    except Exception as e:
        st.error(f"Error loading suggestions CSV file: {e}")
        return None
    

@st.cache_data
def recs_clean(recs, pass_criteria, majority_threshold):
    unique_authors_count = recs["author-id"].nunique() -1
    st.write(f"Number of participants who voted: {unique_authors_count}")
    recs = recs[recs["moderated"]==1]
    cols_to_keep = ["comment-body","agrees","disagrees"]
    if 'Rec #' in recs.columns:
        cols_to_keep.append('Rec #')
    if 'Rationale' in recs.columns:
        cols_to_keep.append('Rationale')
    recs = recs[cols_to_keep]
    recs = recs.rename(columns={'comment-body': 'Recommendation'})
    recs["neutral/pass"] = unique_authors_count - recs["agrees"] - recs["disagrees"]
    if pass_criteria == 'Include':
        recs["approval"] = (recs["agrees"]/unique_authors_count)*100
    else: 
        recs["approval"] = (recs["agrees"]/(recs['agrees']+recs['disagrees']))*100
    
    recs["majority"] = recs['approval'].apply(lambda x: "✅" if x >= majority_threshold else "❌")
    if 'Rec #' in recs.columns:
        recs['Rec #'] = recs['Rec #'].round().astype(int).astype(str)
    return recs

def sort_recommendations(merged_df, sort_criteria):
    if sort_criteria == 'Agreement':
        merged_df = merged_df.sort_values(by="approval", ascending=False)
    elif sort_criteria == 'Disagreement':
        merged_df = merged_df.sort_values(by='disagrees', ascending=False)
    elif sort_criteria == 'Pass':
        merged_df = merged_df.sort_values(by='neutral/pass', ascending=False)
    elif sort_criteria == 'Original Rec #':
        if 'Rec #' in merged_df.columns:
            merged_df = merged_df.sort_values(by='Rec #')
    return merged_df
    
def embed_recommendations(recs, filename, text_column):
    # Load OpenAI API key from environment variables
    openai.api_key = os.getenv("OPENAI_API_KEY")
    
    def normalize_l2(x):
        x = np.array(x)
        if x.ndim == 1:
            norm = np.linalg.norm(x)
            if norm == 0:
                return x
            return x / norm
        else:
            norm = np.linalg.norm(x, 2, axis=1, keepdims=True)
            return np.where(norm == 0, x, x / norm)
    
    def get_embedding(text):
        try:
            client = OpenAI()
            response = client.embeddings.create(
                input=text,
                model="text-embedding-3-small"
            )
            embedding = response.data[0].embedding[:256]  # Normalize to 256 dimensions
            normalized_embedding = normalize_l2(embedding)  # Normalize the embedding
            return normalized_embedding
        except Exception as e:
            st.error(f"Error generating embedding: {e}")
            return None

    # Apply embedding to each recommendation
    with st.spinner('Generating embeddings...'):
        recs['embedding'] = recs[text_column].apply(get_embedding)

    # Cache the updated dataframe
    recs.to_csv(filename, index=False)

    return recs

def k_means_clustering(recs, k):
    # Perform k-means clustering
    with st.spinner('Clustering in progress...'):
        if 'embedding' in recs.columns:
            embeddings = np.vstack(recs['embedding'].values)
            kmeans = KMeans(n_clusters=k, random_state=42)
            recs['cluster'] = kmeans.fit_predict(embeddings)
        else:
            st.error("Embeddings not found in the dataframe for clustering.")
    return recs

def tsne_2d_embeddings(recs,recommendations_df=None):
    if 'embedding' in recs.columns:
        matrix = np.vstack(recs['embedding'].values)
        tsne = TSNE(n_components=2, perplexity=15, random_state=42, init="random", learning_rate=200)
        vis_dims2 = tsne.fit_transform(matrix)
        x = [x for x, y in vis_dims2]
        y = [y for x, y in vis_dims2]

        # Create a DataFrame for easy manipulation
        df = pd.DataFrame({'x': x, 'y': y, 'cluster': recs['cluster'], 'suggestion': recs['suggestion'], 'quote': '"' + recs['quote'] + '"'})
        
        # Ensure the color array matches the length of the DataFrame
        color = df['cluster'].astype(str)
        # Define a custom color sequence excluding red and green to ensure distinctness from recs
        custom_colors = [
            '#636EFA', # purple
            '#FF9616', # orange
            '#19D3F3', # teal
            '#B6E880', # green pale
            '#EEA6FB', #lightpink
            '#FECB52', # pale orange
            '#511CFB', # dark blue
            '#00B5F7', # light blue
            '#C9FBE5', # light mint
            '#86CE00', #lime green 
            '#BC7196', # pale purple dark
            '#7E7DCD', #violet
            '#FE00CE', #hot pink
            '#F6F926', #yellow
            '#FED4C4' # pale peach
        ]
        # Plot using Plotly
        fig = px.scatter(df, x='x', y='y',
                         color=color,
                         color_discrete_sequence=custom_colors,  # Add this line
                         hover_data={'suggestion': True,'quote':True,'x':False,'y':False},
                         title="Clusters identified visualized in language 2D using t-SNE")
        
        fig.update_traces(marker_size=10, marker=dict(line=dict(width=1, color="DarkSlateGrey")))
        
        # Add recommendations separately as a new trace
        if recommendations_df is not None and 'embedding' in recommendations_df.columns:
            # Transform the embeddings in recommendations_df using the existing t-SNE transformation
            rec_matrix = np.vstack(recommendations_df['embedding'].values)
            rec_vis_dims2 = tsne.fit_transform(matrix)[:len(rec_matrix)]  # Use the same t-SNE transformation

            rec_x = [x for x, y in rec_vis_dims2]
            rec_y = [y for x, y in rec_vis_dims2]

            rec_df = pd.DataFrame({
                'x': rec_x, 
                'y': rec_y, 
                'Rec': recommendations_df['Recommendation'],
                'Rationale': recommendations_df['Rationale'],
                'Passed': recommendations_df['majority']
            })
            
            # Create two separate traces for passed and failed recommendations
            passed_mask = rec_df['Passed'] == '✅'
            
            # Trace for passed recommendations
            passed_trace = px.scatter(
                rec_df[passed_mask],
                x='x', y='y',
                hover_data={'Rec': True, 'Rationale': True, 'Passed': True, 'x': False, 'y': False}
            ).data[0]
            passed_trace.marker = dict(symbol='x', size=15, color='green', line=dict(width=2, color="DarkSlateGrey"), opacity=0.8)
            passed_trace.name = 'passed recs'
            passed_trace.showlegend = True
            
            # Trace for failed recommendations
            failed_trace = px.scatter(
                rec_df[~passed_mask],
                x='x', y='y',
                hover_data={'Rec': True, 'Rationale': True, 'Passed': True, 'x': False, 'y': False}
            ).data[0]
            failed_trace.marker = dict(symbol='cross', size=15, color='red', line=dict(width=2, color="DarkSlateGrey"), opacity=0.8)
            failed_trace.name = 'failed recs'
            failed_trace.showlegend = True
            
            fig.add_trace(passed_trace)
            fig.add_trace(failed_trace)

        fig.update_traces(hoverlabel_font_size=18)  # Increase hover text size
        fig.update_layout(height=600)
        st.plotly_chart(fig)

        return np.array(x), np.array(y)
    else:
        st.error("Embeddings not found in the dataframe for t-SNE.")
        return None, None


def tsne_3d_embeddings(recs,recommendations_df=None):
    if 'embedding' in recs.columns:
        matrix = np.vstack(recs['embedding'].values)
        tsne = TSNE(n_components=3, perplexity=15, random_state=42, init="random", learning_rate=200)
        vis_dims3 = tsne.fit_transform(matrix)

        x = [x for x, y, z in vis_dims3]
        y = [y for x, y, z in vis_dims3]
        z = [z for x, y, z in vis_dims3]
        
        # Create a DataFrame for easy manipulation
        df = pd.DataFrame({'x': x, 'y': y, 'z': z, 'cluster': recs['cluster'], 'suggestion': recs['suggestion'], 'quote': '"' + recs['quote'] + '"'})
        
        # Ensure the color array matches the length of the DataFrame
        color = df['cluster'].astype(str)
        # Define a custom color sequence excluding red and green
        custom_colors = [
            '#636EFA', # purple
            '#FF9616', # orange
            '#19D3F3', # teal
            '#B6E880', # green pale
            '#EEA6FB', #lightpink
            '#FECB52', # pale orange
            '#511CFB', # dark blue
            '#00B5F7', # light blue
            '#C9FBE5', # light mint
            '#86CE00', #lime green 
            '#BC7196', # pale purple dark
            '#7E7DCD', #violet
            '#FE00CE', #hot pink
            '#F6F926', #yellow
            '#FED4C4' # pale peach
        ]

        # Plot using Plotly
        fig = px.scatter_3d(df, x='x', y='y', z='z',
                            color=color,
                            color_discrete_sequence=custom_colors,
                            hover_data={'suggestion': True,'quote':True,'cluster': False,'x':False,'y':False,'z':False},
                            title="Clusters identified visualized in language 3D using t-SNE")

        fig.update_traces(marker_size=5, marker=dict(line=dict(width=1, color="DarkSlateGrey")))

        # Add recommendations separately as a new trace
        if recommendations_df is not None and 'embedding' in recommendations_df.columns:
            # Transform the embeddings in recommendations_df using the existing t-SNE transformation
            rec_matrix = np.vstack(recommendations_df['embedding'].values)
            rec_vis_dims3 = tsne.fit_transform(matrix)[:len(rec_matrix)]

            rec_x = [x for x, y, z in rec_vis_dims3]
            rec_y = [y for x, y, z in rec_vis_dims3]
            rec_z = [z for x, y, z in rec_vis_dims3]

            rec_df = pd.DataFrame({
                'x': rec_x,
                'y': rec_y,
                'z': rec_z,
                'Rec': recommendations_df['Recommendation'],
                'Rationale': recommendations_df['Rationale'],
                'Passed': recommendations_df['majority']
            })

            # Create two separate traces for passed and failed recommendations
            passed_mask = rec_df['Passed'] == '✅'

            # Trace for passed recommendations
            passed_trace = go.Scatter3d(
                x=rec_df[passed_mask]['x'],
                y=rec_df[passed_mask]['y'],
                z=rec_df[passed_mask]['z'],
                mode='markers',
                marker=dict(symbol='x', size=7, color='green', line=dict(width=2, color="DarkSlateGrey"), opacity=0.8),
                name='passed recs',
                hovertemplate="<br>".join([
                    "Rec: %{customdata[0]}",
                    "Rationale: %{customdata[1]}",
                    "Passed: %{customdata[2]}"
                ]),
                customdata=rec_df[passed_mask][['Rec', 'Rationale', 'Passed']].values
            )

            # Trace for failed recommendations
            failed_trace = go.Scatter3d(
                x=rec_df[~passed_mask]['x'],
                y=rec_df[~passed_mask]['y'],
                z=rec_df[~passed_mask]['z'],
                mode='markers',
                marker=dict(symbol='cross', size=15, color='red', line=dict(width=2, color="DarkSlateGrey"), opacity=0.8),
                name='failed recs',
                hovertemplate="<br>".join([
                    "Rec: %{customdata[0]}",
                    "Rationale: %{customdata[1]}",
                    "Passed: %{customdata[2]}"
                ]),
                customdata=rec_df[~passed_mask][['Rec', 'Rationale', 'Passed']].values
            )

            fig.add_trace(passed_trace)
            fig.add_trace(failed_trace)

        # Show the plot
        fig.update_traces(hoverlabel_font_size=18)  # Increase hover text size
        fig.update_layout(height=700, scene_camera=dict(eye=dict(x=0.65, y=0.65, z=0.5)))  # Default zoom set higher
        st.plotly_chart(fig)
        return np.array(x), np.array(y), np.array(z)
    else:
        st.error("Embeddings not found in the dataframe for t-SNE.")
        return None, None, None
    
def load_suggestions_embeddings():
    if not os.path.exists('results/all_suggestions_categorized_with_embeddings.csv'):
        recs = load_suggestions()
        if recs is not None:
            recs = embed_recommendations(recs, 'results/all_suggestions_categorized_with_embeddings.csv', 'suggestion')
    else:
        recs = pd.read_csv('results/all_suggestions_categorized_with_embeddings.csv')
        
    if 'embedding' in recs.columns:
        recs['embedding'] = recs['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
        return recs
    else:
        st.error("Embeddings not found in the CSV file.")
        return None

def main_page():
    st.title("Visualizing suggestion clusters")
    
    recs  = load_suggestions_embeddings()
    st.markdown("We've taken all 600+ suggestions and vectorized them using an embedding model. Then, we use k-means clustering to group suggestions by topic, and they are visualized below using t-SNE.")
    st.markdown("You can drag the sliders to toggle between 2 and 3D and change the number of clusters!")

    col1, col2, col3, _ = st.columns([0.2, 0.2, 0.4, 0.2])
    with col1:
        show_recs = st.toggle('Show recommendations', value=True)
    with col2:
        dim = 3 if st.toggle('Plot in 3D', value=False) else 2
    with col3:
        k = st.slider('Select number of clusters (k):', min_value=2, max_value=15, value=5, step=1, label_visibility="collapsed")

    # Perform k-means clustering
    recs = k_means_clustering(recs,k)
    
    
    recommendations_df = None
    if show_recs:
        recommendations_df = load_recommendations()
        recommendations_df = format_recommendations(recommendations_df)
        
    with st.spinner('t-SNE in progress...'):
        if dim==3:
            st.markdown("Hover over the datapoints to see the suggestion and original quote. You can also zoom in/out and rotate the plot.")
            x,y,z = tsne_3d_embeddings(recs, recommendations_df)
        else:
            st.markdown("Hover over the datapoints to see the suggestion and original quote.")
            x,y = tsne_2d_embeddings(recs, recommendations_df)

    
    
def explanation_page():
    st.title("Surfacing suggestions with LLMs")
    st.markdown("We processed the audio from days 1 and 2 of the assembly and used a large language model (LLM, specifically `gpt-4o-mini`) to surface any ideas or suggestions for recommendations in the transcripts. This resulted in a whopping **656 suggestions!**")
    st.markdown("*Note: we only used the breakout sessions 2 and 3 from day 1, and breakout 1 from day 2, i.e. **excludes the specific draft brainstorming sessions.** If you're curious why, ask!*)")
    st.markdown('---')
    st.write("Below is a random sample of 10 AI-paraphrased suggestions along with their original quote. While imperfect, you can get a sense of all the ideas that surfaced in your conversations. Do you recognize any of the quotes?")


    # Check if the CSV file exists
    if os.path.exists('results/all_suggestions_categorized_with_embeddings.csv'):
        # Load the CSV file
        suggestions_df = pd.read_csv('results/all_suggestions_categorized_with_embeddings.csv')

        # Function to sample and display random rows
        def display_random_sample():
            sample_df = suggestions_df[['suggestion', 'quote']].sample(n=10)
            sample_df['quote'] = sample_df['quote'].apply(lambda x: f'"{x}"')
            styled_sample_df = sample_df.style.set_properties(**{
                                    'white-space': 'pre-wrap',  # Allows text to wrap
                                    'word-wrap': 'break-word',  # Breaks long words
                                })
            hide_index_css = """
            <style>
            .level0.row_heading {
                visibility: hidden;
            }
            .custom-table th, .custom-table td {
                width: 50%;  /* Set equal width for columns */
            }
            <</style>
            """
            st.markdown(hide_index_css,unsafe_allow_html=True)
            st.table(styled_sample_df.set_table_styles({
                ('',): [{'selector': 'table', 'props': [('class', 'custom-table')]}]
            }))
            
        # Add a button to generate a new random sample
        if st.button('(re)generate sample'):
            display_random_sample()
        else: 
            display_random_sample()

    else:
        st.error("CSV file not found.")

def format_recommendations(recommendations_df):
    # Rename columns in recommendations_df
    recommendations_df['Recommendation'] = recommendations_df.apply(
        lambda row: row['Recommendation'][:150]+"..." if len(row['Recommendation'])>150 else row['Recommendation'], axis=1
    )
    recommendations_df['Rationale'] = recommendations_df.apply(
        lambda row: row['Rationale'][:150]+"..." if len(row['Rationale'])>150 else row['Rationale'], axis=1
    )
    recommendations_df = recommendations_df[['Recommendation','Rationale', 'majority', 'embedding']]
    
    return recommendations_df

def load_recommendations():
    # Check if the CSV file exists
    if os.path.exists('results/final_votes_with_rationales_embeddings.csv'):
        recommendations_df = pd.read_csv('results/final_votes_with_rationales_embeddings.csv')
        recommendations_df['embedding'] = recommendations_df['embedding'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
        return recommendations_df
    else:
        st.error("CSV file not found.")
        return None


def recommendations_page():
    st.title("Recommendations")
    st.markdown("This page provides a detailed view of the recommendations generated from the assembly discussions.")
    recs = load_suggestions_embeddings()
    recommendations_df = load_recommendations()
    recommendations_df = format_recommendations(recommendations_df)
    recommendations_df


    
    
    

# Add the new page to the pages dictionary
pages = {
    "Results": [
        st.Page(explanation_page, title="Surfacing"),
        st.Page(main_page, title="Clustering"),
        st.Page(recommendations_page, title="Recommendations"),
    ]
}
pg = st.navigation(pages)
pg.run()
