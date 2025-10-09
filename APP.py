# =====================================================
# Hebrew Word Visualization App (Streamlit) - IMPROVED
# Senior Capstone Prototype
# =====================================================

# Description of our Project:
# This Streamlit app allows users to input Hebrew words from a book of the Bible and visualize various aspects of these words.
# Users can see character frequency, word length distribution, interactive word embeddings with clustering, and a co-occurrence network.
# The app includes filtering options for word length and first letter, as well as adjustable parameters for t-SNE and KMeans clustering.

#TODO
#Make my UI very friendly. Make it something more focused on the bible.

#Parameters: Number of similar words per level (numbers). Search Depth (Numbers)

#Remove everything except co occurence networks and word embeddings. Remove filters
#Move boxes over to the left side.

#Send Dr. ALferez pictures of the app.


# --- Imports ---
import streamlit as st
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
import io
import seaborn as sns
from pyvis.network import Network
import streamlit.components.v1 as components
from network_tools import VocabNet
from gensim.models.keyedvectors import KeyedVectors


# =====================================================
# App Configuration
# =====================================================
st.set_page_config(
    page_title="Hebrew Word Explorer",
    page_icon="📊",
    layout="wide"
)

# =====================================================
# App Title and Description
# =====================================================
st.title("📖 Hebrew Word Explorer")
st.markdown("""
Explore how Biblical Hebrew words connect, co-occur, and relate semantically through Scripture.
    <style>
    .block-container {
        padding-top: 1rem;
        padding-bottom: 1rem;
        max-width: 1100px;
        margin: auto;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Instructions
st.markdown("**Instructions:** Paste Hebrew words separated by spaces in the text area below.")

# =====================================================
# Sidebar Configuration
# =====================================================
st.sidebar.header("🎛️ Visualization Options")

# Main visualization choice
visualization_choice = st.sidebar.radio(
    "Choose a visualization:",
    [
        "Co-occurrence Network","Word Embeddings","Advanced Analytics"
    ],
)

if visualization_choice == "Word Embeddings":
    st.sidebar.markdown("### Word Network Settings")
    st.sidebar.caption("Control how deeply and widely the Hebrew word network explores relationships.")
    
    num_similar = st.sidebar.number_input(
        "Number of Similar Words per Level", 
        min_value=1, max_value=100, value=10,
        help="Controls how many related words are added per level."
    )

    search_depth = st.sidebar.number_input(
        "Search Depth", 
        min_value=1, max_value=10, value=2,
        help="Controls how many levels deep the network searches."
    )

if visualization_choice == "🕸️ Co-occurrence Network":
    st.sidebar.markdown("### 🕸️ Network Settings")
    min_edge_weight = st.sidebar.slider(
        "Minimum Edge Weight (filter weak links)",
        min_value=1, max_value=10, value=1
    )
else:
    min_edge_weight = 1  # fallback default

# =====================================================
# User Input Section
# =====================================================
st.subheader("Enter Hebrew Words")

# Sample text option
if st.checkbox("Use sample Hebrew text"):
    sample_text = "בראשית ברא אלהים את השמים ואת הארץ והארץ היתה תהו ובהו וחשך על פני תהום ורוח אלהים מרחפת על פני המים"
    user_input = st.text_area("Hebrew words (sample loaded):", value=sample_text, height=100)
else:
    user_input = st.text_area("Type or paste Hebrew words separated by spaces:", height=100, 
                              placeholder="הקלד כאן מילים בעברית...")

# Allow uploading a text or csv file containing Hebrew words. Clicking a file in the uploader
# will load its contents into the text area and trigger visualizations when the "Word Embeddings"
# option is selected.

# Process the input into a cleaned list of words according to the sidebar filters
raw_words = [] if not user_input else user_input.split()
words = [w.strip() for w in raw_words if w.strip()]

# Inform user when there are no words to analyze
if not words:
    st.info("No words available. Type in words to begin.")

# =====================================================
# Visualization Implementation
# =====================================================

elif visualization_choice == "Character Frequency":
    st.subheader("Character Frequency Analysis")
    
    char_freq = Counter("".join(words))
    total_chars = sum(char_freq.values())
    
    # Create DataFrame
    df_freq = pd.DataFrame({
        "Character": list(char_freq.keys()),
        "Count": list(char_freq.values()),
        "Percentage": [round(v / total_chars * 100, 2) for v in char_freq.values()]
    }).sort_values(by="Count", ascending=False)
    
    # Display table and chart
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.dataframe(df_freq, use_container_width=True)

    if not user_input:
        sample_text = "apple orange banana"
        user_input = sample_text
    
    with col2:
        fig = px.bar(df_freq.head(20), x='Character', y='Count', 
                     title='Top 20 Character Frequencies',
                     labels={'Count': 'Frequency'})
        fig.update_layout(xaxis_title="Hebrew Characters", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

elif visualization_choice == "Word Length Distribution":
    st.subheader("Word Length Distribution")
    
    word_lengths = [len(w) for w in words]
    
    # Create histogram
    fig = px.histogram(x=word_lengths, nbins=max(word_lengths)-min(word_lengths)+1,
                       title="Distribution of Word Lengths",
                       labels={'x': 'Word Length (characters)', 'y': 'Frequency'})
    fig.update_traces(marker_line_width=1, marker_line_color="black")
    st.plotly_chart(fig, use_container_width=True)
    
    # Summary statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Min Length", min(word_lengths))
    with col2:
        st.metric("Max Length", max(word_lengths))
    with col3:
        st.metric("Most Common Length", Counter(word_lengths).most_common(1)[0][0])

elif visualization_choice == "Co-occurrence Network":
    st.subheader("Co-occurrence Network")
    st.info("Network shows word pairs that appear consecutively.")
    
    # Build co-occurrence graph
    cooc = Counter()
    for i in range(len(words)-1):
        cooc[(words[i], words[i+1])] += 1
    
    # Filter by minimum edge weight
    filtered_cooc = {k: v for k, v in cooc.items() if v >= min_edge_weight}
    
    if not filtered_cooc:
        st.warning("No co-occurrences found with the current minimum edge weight. Try reducing the threshold.")
    else:
        G = nx.Graph()
        for (w1, w2), count in filtered_cooc.items():
            G.add_edge(w1, w2, weight=count)
        
        if G.number_of_nodes() > 0:
            # Create layout
            pos = nx.spring_layout(G, k=1, iterations=50)
            
            # Prepare edge traces
            edge_x = []
            edge_y = []
            for edge in G.edges():
                x0, y0 = pos[edge[0]]
                x1, y1 = pos[edge[1]]
                edge_x.extend([x0, x1, None])
                edge_y.extend([y0, y1, None])
            
            # Prepare node traces
            node_x = []
            node_y = []
            node_text = []
            for node in G.nodes():
                x, y = pos[node]
                node_x.append(x)
                node_y.append(y)
                node_text.append(node)
            
            # Create Plotly figure
            fig = go.Figure()
            
            # Add edges
            fig.add_trace(go.Scatter(x=edge_x, y=edge_y,
                                   line=dict(width=2, color='lightgray'),
                                   hoverinfo='none',
                                   mode='lines'))
            
            # Add nodes
            fig.add_trace(go.Scatter(x=node_x, y=node_y,
                                   mode='markers+text',
                                   hoverinfo='text',
                                   text=node_text,
                                   textposition="middle center",
                                   marker=dict(size=20, color='lightblue', 
                                             line=dict(width=2, color='darkblue'))))
            
            fig.update_layout(title="Hebrew Word Co-occurrence Network",
                            showlegend=False,
                            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
                            height=600)
            
            st.plotly_chart(fig, use_container_width=True)
        
        # Display edge weights
        st.subheader("Co-occurrence Frequencies")
        df_cooc = pd.DataFrame([(f"{k[0]} → {k[1]}", v) for k, v in filtered_cooc.items()], 
                              columns=['Word Pair', 'Frequency']).sort_values('Frequency', ascending=False)
        st.dataframe(df_cooc, use_container_width=True)

elif visualization_choice == "Word Embeddings":
        # from st_link_analysis import st_link_analysis

    st.subheader("📚 Word Embeddings Network")

    # Create a directed graph
    dg = nx.DiGraph()

    # Build a NetworkX graph
    # G = nx.erdos_renyi_graph(n=30, p=0.1, seed=42)
    G = VocabNet()
    G.word_search(KeyedVectors.load("stored_nt_vectors.kv"), num_similar, None, search_depth)
    G = G.dg
    net = Network(
    height="800px", width="100%", directed=True,
    bgcolor="#1a1a1a", font_color="#f1f1f1")

    data = nx.node_link_data(G)

    # Display
    # st_link_analysis(data)

    # # Save and embed in Streamlit
    # net.save_graph("directed_graph.html")
    # HtmlFile = open("directed_graph.html", "r", encoding="utf-8")
    # components.html(HtmlFile.read(), height=550)

    print('netexp')
    net = Network(height="1000px", width="100%", directed=True, bgcolor="#000000", font_color="white")

    # Add nodes & edges
    for n, attrs in G.nodes(data=True):
        net.add_node(n, title=n, **attrs)
    for u, v, attrs in G.edges(data=True):
        net.add_edge(u, v, arrows="to", **attrs)


    # # Prep & render
    net.repulsion()
    net.prep_notebook()
    net.show_buttons(filter_=["layout", "interaction", "nodes", "edges"])
    net.show("sim_graph.html")
    HtmlFile = open("sim_graph.html", "r", encoding="utf-8")
    components.html(HtmlFile.read(), height=1000)

    net.toggle_physics(True)

elif visualization_choice == "Advanced Analytics":
    st.subheader("Advanced Analytics")
    
    # Word similarity matrix
    if len(set(words)) >= 2:
        st.subheader("Word Similarity Matrix")
        
        try:
            vectorizer = CountVectorizer(analyzer='char')
            X = vectorizer.fit_transform(list(set(words)))
            similarity_matrix = cosine_similarity(X)
            
            unique_words = list(set(words))
            df_sim = pd.DataFrame(similarity_matrix, 
                                index=unique_words, 
                                columns=unique_words)
            
            # Create heatmap
            fig = px.imshow(df_sim, text_auto='.2f', aspect="auto",
                          title="Word Similarity Heatmap (Character-based)")
            st.plotly_chart(fig, use_container_width=True)
        except:
            st.error("Could not compute similarity matrix")
    
    # Character n-gram analysis
    st.subheader("Character Bigram Analysis")
    try:
        vectorizer_bigram = CountVectorizer(analyzer='char', ngram_range=(2, 2))
        bigram_matrix = vectorizer_bigram.fit_transform(words)
        bigram_freq = dict(zip(vectorizer_bigram.get_feature_names_out(), 
                              bigram_matrix.sum(axis=0).A1))
        
        df_bigram = pd.DataFrame(bigram_freq.items(), 
                                columns=['Bigram', 'Frequency']).sort_values('Frequency', ascending=False)
        
        fig = px.bar(df_bigram.head(20), x='Bigram', y='Frequency',
                    title='Top 20 Character Bigrams')
        st.plotly_chart(fig, use_container_width=True)
    except:
        st.error("Could not compute bigram analysis")

