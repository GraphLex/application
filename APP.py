# =====================================================
# Hebrew Word Visualization App (Streamlit)
# Senior Capstone Prototype
# =====================================================

# Description of our Project:
# This Streamlit app allows users to input Hebrew words from a book of the Bible and visualize various aspects of these words.
# Users can see character frequency, word length distribution, interactive word embeddings with clustering, and a co-occurrence network.
# The app includes filtering options for word length and first letter, as well as adjustable parameters for t-SNE and KMeans clustering.

# --- Imports ---
import streamlit as st
import matplotlib.pyplot as plt
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import networkx as nx

# =====================================================
# App Title
# =====================================================
st.title("Hebrew Word Visualization App (Robust Version)")
st.markdown("""
This app allows you to visualize Hebrew words from a book of the Bible.
Features include character frequency, word length distribution, interactive word embeddings with clustering, and a co-occurrence network.
""")

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
        "Co-occurrence Network",
        "Word Embeddings"
    ]
)

# Filters
min_len, max_len = st.sidebar.slider("Filter by word length", 1, 20, (1, 10))
first_letter = st.sidebar.text_input("Filter by first letter (optional)")

# t-SNE and KMeans options
perplexity = st.sidebar.slider("t-SNE Perplexity", 5, 50, 30)
n_clusters = st.sidebar.slider("KMeans Clusters", 2, 20, 10)

# =====================================================
# User Input
# =====================================================
st.subheader("Enter Hebrew words")
user_input = st.text_area("Type Hebrew words separated by spaces:")

# =====================================================
# Preprocessing Input Words
# =====================================================
if user_input.strip():
    words = user_input.split()
else:
    st.info("Enter some Hebrew words to begin analysis.")
    words = []

# Apply filters
words = [w.strip() for w in words if w.strip()]  # remove empty strings
words = [w for w in words if min_len <= len(w) <= max_len]
if first_letter:
    words = [w for w in words if w.startswith(first_letter)]

# Remove duplicates
words = list(set(words))

if not words:
    st.warning("No words available after filtering.")
    st.stop()

# =====================================================
# Basic Stats Data
# =====================================================
char_freq = Counter("".join(words))
word_lengths = [len(w) for w in words]

# =====================================================
# Visualization Branching
# =====================================================
if visualization_choice == "Basic Stats":
    st.subheader("Basic Word Analysis")
    st.write(f"Total words: {len(words)}")
    st.write(f"Unique words: {len(set(words))}")
    st.write("Words Entered:")
    st.write(words)

elif visualization_choice == "Character Frequency":
    st.subheader("Character Frequency (Hebrew letters)")

    # Raw counts
    st.write("Raw counts:")
    st.write(char_freq)

    # Relative frequency
    total_chars = sum(char_freq.values())
    df_freq = pd.DataFrame({
        "Character": list(char_freq.keys()),
        "Count": list(char_freq.values()),
        "Relative (%)": [round(v / total_chars * 100, 2) for v in char_freq.values()]
    }).sort_values(by="Count", ascending=False)
    st.dataframe(df_freq)

    # Plot
    fig, ax = plt.subplots()
    ax.bar(df_freq["Character"], df_freq["Count"])
    ax.set_title("Hebrew Character Frequency")
    ax.set_xlabel("Character")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif visualization_choice == "Word Length Distribution":
    st.subheader("Word Length Distribution")
    fig, ax = plt.subplots()
    ax.hist(word_lengths, bins=range(1, max(word_lengths)+2), edgecolor='black')
    ax.set_title("Distribution of Word Lengths")
    ax.set_xlabel("Word Length")
    ax.set_ylabel("Frequency")
    st.pyplot(fig)

elif visualization_choice == "Co-occurrence Network":
    st.subheader("Co-occurrence Network")
    st.info("Network shows consecutive word pairs as edges.")

    # Build co-occurrence
    cooc = Counter()
    for i in range(len(words)-1):
        cooc[(words[i], words[i+1])] += 1

    G = nx.Graph()
    for (w1, w2), count in cooc.items():
        G.add_edge(w1, w2, weight=count)

    if G.number_of_nodes() == 0:
        st.warning("Not enough words to create a network.")
    else:
        pos = nx.spring_layout(G)
        plt.figure(figsize=(8, 6))
        nx.draw(G, pos, with_labels=True, node_size=700, node_color="skyblue", font_size=12, width=[G[u][v]['weight'] for u,v in G.edges()])
        st.pyplot(plt)

elif visualization_choice == "Word Embeddings":
    st.subheader("Word Embeddings t-SNE Visualization")

    # Convert words to numerical vectors (character-level)
    vectorizer = CountVectorizer(analyzer='char')
    X = vectorizer.fit_transform(words).toarray()

    # Compute t-SNE embeddings
    tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
    tsne_results = tsne.fit_transform(X)

    
    # # Create DataFrame
    # df_tsne = pd.DataFrame({
    #     'word': words,
    #     'tsne_1': tsne_results[:, 0],
    #     'tsne_2': tsne_results[:, 1]
    # })

    # # KMeans clustering
    # if df_tsne.shape[0] < n_clusters:
    #     st.warning("Number of clusters is larger than number of words. Reducing clusters automatically.")
    #     n_clusters = max(1, df_tsne.shape[0])
    # kmeans = KMeans(n_clusters=n_clusters, random_state=42)
    # df_tsne['cluster'] = kmeans.fit_predict(df_tsne[['tsne_1', 'tsne_2']])

    # Plot interactive Plotly scatter
    fig = px.scatter(
        df_tsne,
        x='tsne_1', y='tsne_2',
        color='cluster',
        hover_data=['word'],
        title=f"t-SNE Word Embeddings with {n_clusters} Clusters"
    )
    st.plotly_chart(fig, use_container_width=True)
