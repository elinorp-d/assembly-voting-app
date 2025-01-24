
import os
import pandas as pd
import numpy as np
import streamlit as st

# SETTING PAGE CONFIG TO WIDE MODE AND ADDING A TITLE AND FAVICON
st.set_page_config(layout="wide", page_title="2025 IAP Student Assembly Voting Results", page_icon=":ballot_box:")
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
    st.markdown("---")
    with st.expander("Upload Data CSV Here:", expanded=True):
        f = st.file_uploader("Data Upload", type="csv", key='recs', label_visibility='hidden')
        if f:
            st.success("File uploaded successfully!")
        # votes = data_upload(st.file_uploader("Upload votes csv", type="csv"))
    

            
@st.cache_data
def recs_clean(recs, sort=None):
    unique_authors_count = recs["author-id"].nunique()
    st.write(f"Number of participants who voted: {unique_authors_count}")
    recs = recs[recs["moderated"]==1]
    recs = recs[["comment-id","comment-body","agrees","disagrees"]]
    recs['comment-id'] +=1
    recs[['Recommendation', 'Rationale']] = recs['comment-body'].apply(lambda x: x.split('Rationale: ', 1) if 'Rationale: ' in x else [x, '']).apply(pd.Series)
    recs["agrees"] -=1
    recs["neutral/pass"] = unique_authors_count - recs["agrees"] - recs["disagrees"]
    recs["approval"] = (recs["agrees"]/unique_authors_count)*100
    recs["majority"] = recs['approval'].apply(lambda x: "✅" if x >= 75 else "❌")

    if sort == 'Agreement':
        recs = recs.sort_values(by="approval", ascending=False)
    elif sort == 'Disagreement':
        recs = recs.sort_values(by='disagrees', ascending=False)
    elif sort == 'Pass':
        recs = recs.sort_values(by='neutral/pass', ascending=False)
    # recs_mapping = {row["comment-id"]:row["comment-body"] for _,row in recs.iterrows()}
    return recs#, recs_mapping

@st.cache_data
def votes_clean(votes, _recs_ids):
    # cols_to_drop = ["group-id","n-comments"]
    # votes = votes.drop(columns=cols_to_drop, errors='ignore')
    comment_ids = list(map(str, _recs_ids.keys()))
    votes = votes[votes.columns.intersection(comment_ids)]
    votes.columns = [f"Recommendation {col}" for col in votes.columns]
    votes = votes.drop(index=0, errors='ignore')
    return votes
    
def main_page():
    st.title("2025 IAP Student Assembly Voting Overall Results")
    recs = data_load(st.session_state.recs)
    if recs is not None:
        # recs, recs_mapping = recs_clean(recs, chosen)
        recs = recs_clean(recs, chosen)
        # st.dataframe(recs,
        #             column_order=['comment-id','Recommendation','agrees','disagrees','neutral/pass','approval','majority'],
        #             column_config={
        #                 "comment-id": "Rec #",
        #                 "Recommendation": "Recommendation",
        #                 # "comment-body": "Recommendation",
        #                 "approval":
        #                     st.column_config.ProgressColumn(
        #                         "% Approval",
        #                         help="% of agree votes",
        #                         format="%.2f %%",
        #                         min_value=0,
        #                         max_value=100,
        #                 ),
        #                     "majority":  "Majority?",
        #             },
        #             hide_index=True,
        #             height=(len(recs) * 35) + 50)  # Adjust height based on number of rows

        recs_formatted = recs[['comment-id', 'Recommendation', 'approval', 'majority']]
        recs_formatted = recs_formatted.rename(columns={
            'comment-id': 'Rec #',
            'approval': '% Approval',
            'majority': 'Majority?'
        })
        
        # Format the approval column as a percentage
        recs_formatted['% Approval'] = recs_formatted['% Approval'].apply(lambda x: f"{x:.2f}%")
        # recs_formatted = recs_formatted.reset_index(drop=True)
        
                # Define a function to highlight cells
        def highlight_row(row):
            try:
                # Remove the percentage sign and convert to float
                value = float(row['% Approval'].strip('%'))
                # Apply background color to the entire row based on the value
                if value >= 75:
                    return ['background-color: lightgreen' for _ in row]
                elif value >= 50:
                    return ['background-color: #DBF9DB' for _ in row]
                
                return ['background-color: lightgreen' if value >= 75 else 'background-color: #DBF9DB' if 50 <= value < 75 else '' for _ in row]
            except ValueError:
                return ['' for _ in row]

        
        styled_recs = recs_formatted.style.apply(highlight_row, axis=1) \
                                .set_properties(**{
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
        # st.table(styled_recs)
        st.table(styled_recs.set_table_styles({
            ('',): [{'selector': 'table', 'props': [('class', 'custom-table')]}]
        }))
    else:
        st.write("Please upload recommendations csv on the left!")

def details_page():
    st.title("2025 IAP Student Assembly Detailed Results")
    recs = data_load(st.session_state.recs)
    if recs is not None:
        recs = recs_clean(recs, chosen)
        for index, row in recs.iterrows():
            container = st.container()
            container.markdown("---")
            col1, col2 = container.columns(2)
            col1.markdown(f"### Recommendation {row['comment-id']}")
            
            # Print the recommendation text
            if row['Recommendation']:
                col1.markdown(f"#### {row['Recommendation']}")
            else:
                col1.markdown(f"#### {row['comment-body']}")
            if row['Rationale']:
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
        

# def votes_page():
    # if votes is not None:
    #     votes = votes_clean(votes, recs_mapping)
    #     st.dataframe(votes)
    #     st.dataframe(votes.style.highlight_max(axis=0))

pages = {
    "Results": [
        st.Page(main_page, title="Overall Results"),
        st.Page(details_page, title="Detailed Results")
    ]
}
pg = st.navigation(pages)
pg.run()
