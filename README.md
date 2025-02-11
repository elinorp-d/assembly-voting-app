# student assembly web apps

I've created two web apps for the student assembly 2025. The first is the voting results visualization, and the second is the clustering visualization of suggestions processed from the first two days of the assembly.

## 1. voting app

This repository contains basic code for importing the Student Assembly votes from Polis and a csv containing the full recommendation texts to display nicely in a Streamlit app. 

The app will allow you to upload two files (see [Data](###data) below) and then display the overall results in a table. Recommendations passing with >=75% are highlighted in green, and recommendations that did not pass but achieved at least 50% are lighter green. The Detailed results shows the recommendations with their associated rationales and a pie chart visualizing the vote breakdown. The sidebar allows for sorting by agreement, disagreement, number of passes, or taking the original order for both pages.

### data

You'll need at least the comments csv from Polis, and optionally also the rationales csv. These can be found in the `data/` folder. If you'd like to use your own, you can directly use any Polis comments csv. The rationales csv contains the correct metadata for each recommendation: the rec number, text, and rationale. The script handles merging these files based on the recommendation text.


## 2. clustering app
I've also created a streamlit app for visualizing the suggestions from Days 1 + 2 and overlaying the Day 2 recommendations (and soon, final recs) onto it. You don't need to upload anything because all the data is in this repository.


## setup / requirements instructions

1. Clone this repository
2. `pip install -r requirements.txt`
3. To start the app, run `streamlit run <filename>.py` in your terminal and it will open in your local browser




