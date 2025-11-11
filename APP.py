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
from collections import Counter
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import plotly.express as px
import plotly.graph_objects as go
import networkx as nx
from sklearn.metrics.pairwise import cosine_similarity
import warnings
warnings.filterwarnings('ignore')
from pyvis.network import Network
import streamlit.components.v1 as components
from network_tools import VocabNet
from gensim.models.keyedvectors import KeyedVectors


# =====================================================
# Bible Book Structure Dictionary
# =====================================================
BIBLE_BOOKS = {
    # Old Testament
    "Genesis": "OT",
    "Exodus": "OT",
    "Leviticus": "OT",
    "Numbers": "OT",
    "Deuteronomy": "OT",
    "Joshua": "OT",
    "Judges": "OT",
    "Ruth": "OT",
    "1 Samuel": "OT",
    "2 Samuel": "OT",
    "1 Kings": "OT",
    "2 Kings": "OT",
    "1 Chronicles": "OT",
    "2 Chronicles": "OT",
    "Ezra": "OT",
    "Nehemiah": "OT",
    "Esther": "OT",
    "Job": "OT",
    "Psalms": "OT",
    "Proverbs": "OT",
    "Ecclesiastes": "OT",
    "Song of Solomon": "OT",
    "Isaiah": "OT",
    "Jeremiah": "OT",
    "Lamentations": "OT",
    "Ezekiel": "OT",
    "Daniel": "OT",
    "Hosea": "OT",
    "Joel": "OT",
    "Amos": "OT",
    "Obadiah": "OT",
    "Jonah": "OT",
    "Micah": "OT",
    "Nahum": "OT",
    "Habakkuk": "OT",
    "Zephaniah": "OT",
    "Haggai": "OT",
    "Zechariah": "OT",
    "Malachi": "OT",
    # New Testament
    "Matthew": "NT",
    "Mark": "NT",
    "Luke": "NT",
    "John": "NT",
    "Acts": "NT",
    "Romans": "NT",
    "1 Corinthians": "NT",
    "2 Corinthians": "NT",
    "Galatians": "NT",
    "Ephesians": "NT",
    "Philippians": "NT",
    "Colossians": "NT",
    "1 Thessalonians": "NT",
    "2 Thessalonians": "NT",
    "1 Timothy": "NT",
    "2 Timothy": "NT",
    "Titus": "NT",
    "Philemon": "NT",
    "Hebrews": "NT",
    "James": "NT",
    "1 Peter": "NT",
    "2 Peter": "NT",
    "1 John": "NT",
    "2 John": "NT",
    "3 John": "NT",
    "Jude": "NT",
    "Revelation": "NT"
}


def safe_load_keyedvectors(path: str):
    """Try to load keyed vectors from `path`. If the file doesn't exist or loading fails,
    show a helpful Streamlit message and return None.
    """
    try:
        return KeyedVectors.load(path)
    except FileNotFoundError:
        st.error(
            f"Word vector file not found: '{path}'.\n\nPlace the file in the app folder or update the path.\n" \
            "If you don't have the vectors, you can generate/load them separately or upload a compatible KeyedVectors file."
        )
        st.stop()
        return None
    except Exception as e:
        st.error(f"Failed to load word vectors from '{path}': {e}")
        st.stop()
        return None


def display_selected_books():
    """Display the currently selected Bible books"""
    if 'selected_books' in st.session_state and st.session_state['selected_books']:
        books = st.session_state['selected_books']
        st.info(f"📖 Selected Books: **{', '.join(books)}**")
        return books
    return None


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
st.markdown(
    """
    <style>
    /* Stronger top-of-page layout: remove top padding and tighten margins */
    .reportview-container .main .block-container,
    .stApp .main .block-container,
    section.main > div.block-container {
        padding-top: 0rem !important;
        padding-bottom: 0.5rem !important;
        max-width: 1100px;
        margin: 0 auto;
    }

    /* Reduce header margins and ensure title sits at the very top */
    h1 { margin-top: 0rem !important; margin-bottom: 0.25rem !important; font-size:48px; }
    .top-intro { margin-top: 0 !important; color: #DDDDDD; font-size:16px; }
    </style>

    <h1>📖 Hebrew Word Explorer</h1>
    <p class="top-intro">Explore how Biblical Hebrew words connect, co-occur, and relate semantically through Scripture.</p>
    """,
    unsafe_allow_html=True,
)

# Instructions
st.markdown("**Instructions:** Select Bible books from the sidebar to train the Word2Vec model and explore Hebrew word relationships.")

# Display currently selected books
current_books = display_selected_books()

# =====================================================
# Sidebar Configuration
# =====================================================
st.sidebar.header("🎛️ Visualization Options")

# =====================================================
# Bible Book Selector (Multi-select)
# =====================================================
st.sidebar.markdown("### 📖 Bible Book Selection")
st.sidebar.caption("Select one or more books to analyze")

selected_books = st.sidebar.multiselect(
    "Choose Bible Books",
    options=list(BIBLE_BOOKS.keys()),
    default=["Genesis"],
    help="Select multiple books to train the Word2Vec model on their combined text"
)

# Load books button
if st.sidebar.button("📥 Load Selected Books", type="primary"):
    if selected_books:
        st.session_state['selected_books'] = selected_books
        st.sidebar.success(f"✅ Loaded {len(selected_books)} book(s)")
        # Here you would trigger your backend Word2Vec retraining
        # For example: retrain_word2vec(selected_books)
    else:
        st.sidebar.warning("⚠️ Please select at least one book")

st.sidebar.markdown("---")

# Main visualization choice
visualization_choice = st.sidebar.radio(
    "Choose a visualization:",
    [
        "Co-occurrence Network",
        "Word Embeddings",
        "Character Frequency",
        "Word Length Distribution",
        "Advanced Analytics"
    ],
)

# =====================================================
# Word Embeddings Sidebar Options
# =====================================================
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

    user_word = st.sidebar.text_input("Enter the word or number")

    if st.sidebar.button("Generate Word Embedding"):
        if not user_word:
            st.warning("Please enter a word or number to generate the network.")
        else:
            # Check if books are selected
            if 'selected_books' not in st.session_state or not st.session_state['selected_books']:
                st.warning("⚠️ Please select and load Bible books first!")
            else:
                st.info(f"Building network for '{user_word}' with {num_similar} similar words per level and depth of {search_depth}.")
                st.info(f"Using books: {', '.join(st.session_state['selected_books'])}")

                if user_word.isdigit():
                    user_word = int(user_word)

                # Load the Word2Vec model (you'll need to modify this to load based on selected books)
                # For example: model_path = f"vectors_{'_'.join(st.session_state['selected_books'])}.kv"
                model = safe_load_keyedvectors("stored_bhsa_vectors.kv")
                
                if model is not None:
                    G = VocabNet()
                    G.word_search(vecs=model, word=user_word, num_steps=search_depth, words_per_level=num_similar)

                    nx_graph = G.dg

                    net = Network(height="800px", width="100%", directed=True,
                                bgcolor="#000000", font_color="#ffffff")
                    
                    for n, attrs in nx_graph.nodes(data=True):
                        net.add_node(n, title=n, **attrs)
                        
                    for u, v, attrs in nx_graph.edges(data=True):
                        net.add_edge(u, v)

                    net.repulsion()
                    net.prep_notebook()
                    net.show("sim_graph.html")
                    HtmlFile = open("sim_graph.html", "r", encoding="utf-8")
                    components.html(HtmlFile.read(), height=800)

# =====================================================
# Co-occurrence Network Sidebar Options
# =====================================================
if visualization_choice == "Co-occurrence Network":
    st.sidebar.markdown("### 🕸️ Network Settings")
    min_edge_weight = st.sidebar.slider(
        "Minimum Edge Weight (filter weak links)",
        min_value=1, max_value=10, value=1
    )
else:
    min_edge_weight = 1

# =====================================================
# User Input Area (Main Panel)
# =====================================================
user_input = st.text_area(
    "Hebrew Text Input",
    height=150,
    placeholder="Paste Hebrew words here (space-separated)...",
    label_visibility="collapsed"
)

# Split on whitespace and remove empty tokens
raw_words = [] if not user_input else [w.strip() for w in user_input.split() if w.strip()]
words = raw_words

# Inform user when there are no words to analyze
if not words and visualization_choice not in ["Word Embeddings"]:
    st.info("No words available. Paste text to begin analysis.")

# =====================================================
# Visualization Implementation
# =====================================================

if visualization_choice == "Character Frequency" and words:
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
    
    with col2:
        fig = px.bar(df_freq.head(20), x='Character', y='Count', 
                     title='Top 20 Character Frequencies',
                     labels={'Count': 'Frequency'})
        fig.update_layout(xaxis_title="Hebrew Characters", yaxis_title="Frequency")
        st.plotly_chart(fig, use_container_width=True)

elif visualization_choice == "Word Length Distribution" and words:
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

elif visualization_choice == "Co-occurrence Network" and words:
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
    st.subheader("📚 Word Embeddings Network")
    
    if 'selected_books' not in st.session_state or not st.session_state['selected_books']:
        st.warning("⚠️ Please select and load Bible books from the sidebar first!")
        st.info("The Word2Vec model will be trained on the selected books to generate word embeddings.")
    else:
        st.info("👈 Enter a word or number in the sidebar and click 'Generate Word Embedding' to visualize the network.")

elif visualization_choice == "Advanced Analytics" and words:
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

# =====================================================
# Footer Information
# =====================================================
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info(
    "This tool analyzes Hebrew words from Biblical texts using Word2Vec embeddings "
    "and various text analysis techniques. Select books to train a custom model on specific Biblical content."
)
