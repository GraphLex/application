# --- Imports ---
import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter
import pandas as pd

#TODO Things to work on and improve.
#Create Word2Vec in the app. Include some numerical values
#A graph or something of distances.
#A controller that indicates which words are the closest
#Type the word in english and Hebrew. Transliterating between the alphabet
#Which other visualizations are useful for scholars?
#Use pandas
#Look for software and tools and apps. Look through bible tools and 
#Set up a toggle bar to where we can see different visualizations: coccurence network, word embedding
#Increase complexity
#Show what is relevant. Consider which functions are relevant for Hebrew

# =====================================================
# Hebrew Word Visualization App
# Senior Capstone Prototype
# =====================================================

# --- App Title ---
st.title("Hebrew Word Visualization App (Working Prototype)")

# --- User Input Section ---
st.subheader("Enter Hebrew words")
user_input = st.text_input("Type Hebrew words separated by spaces:")

# --- Sidebar Toggle for Visualization Options ---
st.sidebar.header("Visualization Options")
visualization_choice = st.sidebar.radio(
    "Choose a visualization:",
    [
        "Basic Stats", 
        "Character Frequency", 
        "Word Length Distribution", 
        "Co-occurrence Network (Coming Soon)", 
        "Word Embeddings (Coming Soon)"
    ]
)

# =====================================================
# If user has entered words, process them
# =====================================================
if user_input:
    # Split user input into a list of words
    words = user_input.split()

    # --- Precompute Common Data ---
    chars = "".join(words)                         # Concatenate all words into a single string
    char_freq = Counter(chars)                     # Count frequency of each character
    word_lengths = [len(w) for w in words]         # List of word lengths
    df_words = pd.DataFrame({                      # DataFrame for word analysis
        "Word": words, 
        "Length": word_lengths
    })

    # =====================================================
    # Visualization Branching Logic
    # =====================================================

    # --- 1. Basic Stats ---
    if visualization_choice == "Basic Stats":
        st.subheader("Basic Word Analysis")
        st.write(f"Total words: {len(words)}")
        st.write(f"Unique words: {len(set(words))}")
        st.write("Words Entered:")
        st.write(words)

    # --- 2. Character Frequency ---
    elif visualization_choice == "Character Frequency":
        st.subheader("Character Frequency (Hebrew letters)")
        st.write(char_freq)  # Show raw counts

        # Plot with Matplotlib
        fig, ax = plt.subplots()
        ax.bar(char_freq.keys(), char_freq.values())
        ax.set_title("Hebrew Character Frequency")
        st.pyplot(fig)

    # --- 3. Word Length Distribution ---
    elif visualization_choice == "Word Length Distribution":
        st.subheader("Distribution of Word Lengths")

        # Plot with Plotly
        fig2 = px.histogram(
            df_words, 
            x="Length", 
            nbins=10, 
            title="Word Length Distribution"
        )
        st.plotly_chart(fig2)

    # --- 4. Placeholder for Co-occurrence Network ---
    elif visualization_choice == "Co-occurrence Network (Coming Soon)":
        st.subheader("Co-occurrence Network")
        st.info("This feature will display a network graph of words that frequently occur together. (Work in Progress)")

    # --- 5. Placeholder for Word Embeddings ---
    elif visualization_choice == "Word Embeddings (Coming Soon)":
        st.subheader("Word Embeddings Visualization")
        st.info("This feature will show Word2Vec embeddings and similarity plots. (Work in Progress)")

# =====================================================
# If no input is given yet
# =====================================================
else:
    st.info("Enter some Hebrew words to begin analysis.")
