#Imports 
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd

# =====================================================
# Hebrew Word Visualization App
# Senior Capstone Prototype
# =====================================================

#TODO: Get reliable results. App allows people to select different approach, either PCA or the other option

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
        "Word Embeddings"
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

        # Also show table of counts
        df_freq = pd.DataFrame(list(char_freq.items()), columns=["Character", "Frequency"])
        st.dataframe(df_freq)

    # --- 3. Word Embeddenings ---
    # Plot the result of our TSNE with the label color coded
    # A lot of the stuff here is about making the plot look pretty and not TSNE
    # Load your TSNE data
tsne_result_df = pd.read_csv("tsne.csv", sep="\t", header=None, names=['tsne_1', 'tsne_2'])

# If you have labels (for example, in a list y), you can add them:
# tsne_result_df['label'] = y
# Otherwise, create a dummy label to make plotting easier
tsne_result_df['label'] = 0  # all same label for now

# Plot
fig, ax = plt.subplots(1, figsize=(8,8))
sns.scatterplot(
    x='tsne_1', 
    y='tsne_2', 
    hue='label', 
    data=tsne_result_df, 
    ax=ax, 
    s=120,
    palette='viridis'  # optional color palette
)

# Set limits and aspect ratio
lim = (tsne_result_df[['tsne_1','tsne_2']].min().min()-5, 
       tsne_result_df[['tsne_1','tsne_2']].max().max()+5)
ax.set_xlim(lim)
ax.set_ylim(lim)
ax.set_aspect('equal')
ax.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.0)

plt.show()

    # --- 4. Placeholder for Co-occurrence Network ---
    elif visualization_choice == "Co-occurrence Network (Coming Soon)":
        st.subheader("Co-occurrence Network")
        st.info("This feature will display a network graph of words that frequently occur together. (Work in Progress)")

# =====================================================
# If no input is given yet
# =====================================================
else:
    st.info("Enter some Hebrew words to begin analysis.")
