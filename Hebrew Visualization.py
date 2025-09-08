import streamlit as st
import matplotlib.pyplot as plt
import plotly.express as px
from collections import Counter

# --- App Title ---
st.title("Hebrew Word Visualization App (Working Prototype)")

# --- User Input ---
st.subheader("Enter Hebrew words")
user_input = st.text_area("Type Hebrew words separated by spaces:")

if user_input:
    # Split words
    words = user_input.split()
    
    # --- Basic Stats ---
    st.subheader("Basic Word Analysis")
    st.write(f"Total words: {len(words)}")
    st.write(f"Unique words: {len(set(words))}")

    # Show list of words
    st.write("Words Entered:")
    st.write(words)

    # --- Character Frequency ---
    chars = "".join(words)
    char_freq = Counter(chars)

    st.subheader("Character Frequency (Hebrew letters)")
    st.write(char_freq)

    # --- Matplotlib Bar Chart ---
    fig, ax = plt.subplots()
    ax.bar(char_freq.keys(), char_freq.values())
    ax.set_title("Hebrew Character Frequency")
    st.pyplot(fig)

    # --- Plotly Word Length Distribution ---
    word_lengths = [len(w) for w in words]
    fig2 = px.histogram(word_lengths, nbins=10, title="Word Length Distribution")
    st.plotly_chart(fig2)

else:
    st.info("Enter some Hebrew words to begin analysis.")