# =====================================================
# Hebrew Word Visualization App (Streamlit)
# Senior Capstone Prototype
# =====================================================

# --- Imports ---
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.cluster import KMeans
import plotly.express as px

# =====================================================
# App Title
# =====================================================
st.title("Hebrew Word Visualization App (Working Prototype)")

# =====================================================
# Sidebar Options
# =====================================================
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
# User Input
# =====================================================
st.subheader("Enter Hebrew words")
user_input = st.text_input("Type Hebrew words separated by spaces:")

# =====================================================
# Function to Plot TSNE with Matplotlib
# =====================================================
def plot_tsne(df, label_col='label'):
    fig, ax = plt.subplots(figsize=(8, 8))
    labels = df[label_col].unique()
    
    # Plot each label separately
    for lbl in labels:
        subset = df[df[label_col] == lbl]
        ax.scatter(subset['tsne_1'], subset['tsne_2'], s=120, label=f"Label {lbl}")
    
    # Set limits and equal aspect ratio
    lim = (
        df[['tsne_1', 'tsne_2']].min().min() - 5,
        df[['tsne_1', 'tsne_2']].max().max() + 5
    )
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_aspect('equal')
    ax.legend(bbox_to_anchor=(1.05, 1), loc=2)
    ax.set_title("t-SNE Visualization")
    st.pyplot(fig)

# =====================================================
# Main Processing Logic
# =====================================================
if user_input:
    # --- Split input and precompute data ---
    words = user_input.split()
    chars = "".join(words)
    char_freq = Counter(chars)
    word_lengths = [len(w) for w in words]
    df_words = pd.DataFrame({
        "Word": words,
        "Length": word_lengths
    })

    # --- Visualization Branching ---
    if visualization_choice == "Basic Stats":
        st.subheader("Basic Word Analysis")
        st.write(f"Total words: {len(words)}")
        st.write(f"Unique words: {len(set(words))}")
        st.write("Words Entered:")
        st.write(words)

    elif visualization_choice == "Character Frequency":
        st.subheader("Character Frequency (Hebrew letters)")
        st.write(char_freq)  # Show raw counts

        # Plot character frequency
        fig, ax = plt.subplots()
        ax.bar(char_freq.keys(), char_freq.values())
        ax.set_title("Hebrew Character Frequency")
        ax.set_xlabel("Character")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

        # Display as a table
        df_freq = pd.DataFrame(list(char_freq.items()), columns=["Character", "Frequency"])
        st.dataframe(df_freq)

    elif visualization_choice == "Word Length Distribution":
        st.subheader("Word Length Distribution")
        fig, ax = plt.subplots()
        ax.hist(word_lengths, bins=range(1, max(word_lengths)+2), edgecolor='black')
        ax.set_title("Distribution of Word Lengths")
        ax.set_xlabel("Word Length")
        ax.set_ylabel("Frequency")
        st.pyplot(fig)

    elif visualization_choice == "Co-occurrence Network (Coming Soon)":
        st.subheader("Co-occurrence Network")
        st.info("This feature will display a network graph of words that frequently occur together. (Work in Progress)")

    elif visualization_choice == "Word Embeddings":
        st.subheader("Word Embeddings t-SNE Visualization")

        try:
            
            df = pd.read_csv("tsne.csv", header =None, names = ['tsne_1', 'tsne_2'])

            df['label'] = 0

                # --- Optional Clustering ---
            kmeans = KMeans(n_clusters=10, random_state=42)
            df['cluster'] = kmeans.fit_predict(df[['tsne']])

            # Plot with Plotly (better interactivity than Matplotlib)
            fig = px.scatter(
                df,
                x="tsne_1", y="tsne_2",
                color="cluster",
                hover_data=["word"]
            )
            st.plotly_chart(fig, use_container_width=True)
                # Plot with Matplotlib (if needed)
                # plot_tsne(df, label_col='label')

        except FileNotFoundError:
            st.error("tsne.csv not found. Make sure the file is in the same directory.")

else:
    st.info("Enter some Hebrew words to begin analysis.")

# Example of adding clustering to the t-SNE plot (if needed)
# Assuming df is the DataFrame with t-SNE results
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(df[['tsne_1','tsne_2']])

df = pd.read_csv('tsne.csv')


fig = px.scatter(
    df,
    x="tsne_1", y="tsne_2",
    color="cluster",
    hover_data=["word"]
)
st.plotly_chart(fig, use_container_width=True)
