
import pandas as pd
import numpy as np
import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="2025 IAP Student Assembly Voting Results Day 3", page_icon=":ballot_box:")
from plotly import graph_objects as go


@st.cache_data
def data_load(uploaded_file):
    if uploaded_file is not None:
        try:
            data = pd.read_csv(uploaded_file)
            return data
        except Exception as e:
            st.error(f"Error loading Recommendations CSV file: {e}")


with st.sidebar:
    st.markdown("#### Options")
    chosen = st.radio(
        'Sort by:',
        ("Agreement", "Disagreement", "Pass", "Original Rec #"),
        key='sort_by')
    passes = st.radio(
        'Include passes when calculating majority?:',
        ("Include", "Exclude"),
        key='passes'
    )
    majority = st.select_slider(
        'Select majority threshold:',
        options=[75, 80, 85],
        value=80,
        key='majority_threshold'
    )
    st.markdown("---")
    with st.expander("Upload Data CSVs Here:", expanded=True):
        f = st.file_uploader("Data Upload", type="csv", key='recs')
        if f:
            st.success("File uploaded successfully!")
        ff = st.file_uploader("Rationales Upload", type="csv", key='rationales')
        if ff:
            st.success("File uploaded successfully!")
    

            
@st.cache_data
def recs_clean(recs, pass_criteria, majority_threshold):
    unique_authors_count = 18
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

@st.cache_data
def rationales_merge(recs, rationales):
    merged_df = recs.merge(rationales, on='Recommendation', how='left')
    return merged_df

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


    
def main_page():
    st.title("2025 IAP Student Assembly Voting Overall Results Day 3")
    recs = data_load(st.session_state.recs)
    # rationales = data_load(st.session_state.rationales)
    if recs is not None:
        recs = recs_clean(recs, passes, majority)
        # if rationales is not None:
        #     recs = rationales_merge(recs,rationales)
        recs = sort_recommendations(recs, chosen)
        if 'Rec #' in recs.columns:
            recs_formatted = recs[['Rec #','Recommendation', 'approval', 'majority']]
        else: 
            recs_formatted = recs[['Recommendation', 'approval', 'majority']]
        recs_formatted = recs_formatted.rename(columns={
            'approval': '% Approval',
            'majority': 'Majority?'
        })
        
        # Format the approval column as a percentage
        recs_formatted['% Approval'] = recs_formatted['% Approval'].apply(lambda x: f"{x:.2f}%") 
        if 'Rec #' in recs_formatted.columns:
            recs_formatted['Rec #'] = recs_formatted['Rec #'].round()

        def highlight_row(row, majority_threshold):
            try:
                # Remove the percentage sign and convert to float
                value = float(row['% Approval'].strip('%'))
                # Apply background color to the entire row based on the value
                if value >= majority_threshold:
                    return ['background-color: lightgreen' for _ in row]
                elif value >= 50:
                    return ['background-color: #DBF9DB' for _ in row]
                
                return ['background-color: lightgreen' if value >= majority_threshold else 'background-color: #DBF9DB' if 50 <= value < majority_threshold else '' for _ in row]
            except ValueError:
                return ['' for _ in row]

        
        styled_recs = recs_formatted.style.apply(highlight_row, axis=1, majority_threshold=majority).set_properties(**{
                                    'white-space': 'pre-wrap',  # Allows text to wrap
                                    'word-wrap': 'break-word',  # Breaks long words
                                })
        hide_index_css = """
        <style>
        .level0.row_heading {
            visibility: hidden;
        }
        <</style>
        """
        st.markdown(hide_index_css,unsafe_allow_html=True)
        st.table(styled_recs.set_table_styles({
            ('',): [{'selector': 'table', 'props': [('class', 'custom-table')]}]
        }))
    else:
        st.write("Please upload recommendations csv on the left!")

def details_page():
    st.title("2025 IAP Student Assembly Detailed Results")
    recs = data_load(st.session_state.recs)
    rationales = data_load(st.session_state.rationales)
    if recs is not None:
        recs = recs_clean(recs, passes, majority)
        if rationales is not None:
            recs = rationales_merge(recs,rationales)
        recs = sort_recommendations(recs, chosen)
        recs.to_csv('results/merged_results.csv')
        for index, row in recs.iterrows():
            container = st.container()
            container.markdown("---")
            col1, col2 = container.columns(2)
            if 'Rec #' in recs.columns:
                col1.markdown(f"### Recommendation {row['Rec #']}")
            else: 
                col1.markdown("### Recommendation")
            
            # Print the recommendation text
            if row['Recommendation']:
                col1.markdown(f"#### {row['Recommendation']}")
            else:
                col1.markdown(f"#### {row['comment-body']}")
            if 'Rationale' in recs.columns:
                col1.markdown(f"*Rationale*: {row['Rationale']}")
            
            # Calculate the percentages for agree, disagree, and neutral/pass
            agree = row['agrees']
            disagree = row['disagrees']
            neutral = row['neutral/pass']
            # Create a donut chart using Plotly
            fig = go.Figure(data=[go.Pie(labels=['Agree', 'Disagree', 'Neutral/Pass'],
                                        values=[agree, disagree, neutral],
                                        hole=.3,
                                        marker_colors=['green', 'red', 'lightgrey'])],
                            
                            )

            # Display the donut chart with a unique key
            col2.plotly_chart(fig, use_container_width=True,key=f"donut_chart_{index}")
            
    else:
        st.write("Please upload recommendations csv on the left!")
        
pages = {
    "Results": [
        st.Page(main_page, title="Overall Results"),
        st.Page(details_page, title="Detailed Results")
    ]
}
pg = st.navigation(pages)
pg.run()
