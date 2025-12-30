# =====================================================
# Hebrew Word Visualization App (Streamlit)
# Senior Capstone Prototype
# =====================================================

# Description of our Project:
# This Streamlit app allows users to input Hebrew words from a book of the Bible and visualize various aspects of these words.
# Users can see character frequency, word length distribution, interactive word embeddings with clustering, and a co-occurrence network.
# The app includes filtering options for word length and first letter, as well as adjustable parameters for t-SNE and KMeans clustering.

#Take away the text box in the beginning when opening the page. Take out the whole radio menu and just leave word 
#Implement function from picture Rhys showed me

#TO DO for next time
# : Go deeper into a particular word. Having a text box for specific word. Have a specific word where you can type it in.
#Reimplement text box where you type in a specific hebrew word.
#Paper: Target audience for the Paper. Next version of paper should have theoretical framework. 
#Explaining the concepts. Have a concept map. Network or graph or semantical analysis. 2-3 concepts explained.

# --- Imports ---
import streamlit as st
import networkx as nx
import warnings
warnings.filterwarnings('ignore')
from pyvis.network import Network
import streamlit.components.v1 as components
from network_tools import NetBuilder, Algorithm
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

#------------------------------------------------------
#About section
#------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 About")
st.sidebar.info(
    "This tool analyzes Hebrew words from Biblical texts using Word2Vec embeddings "
    "and various text analysis techniques. Select books to train a custom model on specific Biblical content."
)

# =====================================================
# Word Embeddings Display
# =====================================================
st.subheader("📚 Word Embeddings Network")

visualization_choice = "Word Embeddings"

if 'selected_books' not in st.session_state or not st.session_state['selected_books']:
    st.warning("⚠️ Please select and load Bible books from the sidebar first!")
    st.info("The Word2Vec model will be trained on the selected books to generate word embeddings.")
elif 'network_generated' in st.session_state and st.session_state['network_generated']:
    # Display the network without regenerating
    components.html(st.session_state['network_html'], height=800)
else:
    st.info("👈 Enter a word or number in the sidebar and click 'Generate Word Embedding' to visualize the network.")

# =====================================================
# Word Embeddings Sidebar Options
# =====================================================
st.sidebar.markdown("### Word Network Settings")
st.sidebar.caption("Control how deeply and widely the Hebrew word network explores relationships.")


 # Input type selection (Word or Strong's Number)
input_type = st.sidebar.radio(
    "Input Type",
    options=["Word", "Strong's Number"],
    horizontal=True
)

if input_type == "Word":
    user_word = st.sidebar.text_input("Enter Hebrew word")
else:
    # Strong's Number input with H/G prefix buttons
    col1, col2, col3 = st.sidebar.columns([1, 1, 3])
    
    with col1:
        if st.button("H", key="hebrew_btn", help="Hebrew Strong's"):
            st.session_state["strongs_prefix"] = "H"
    
    with col2:
        if st.button("G", key="greek_btn", help="Greek Strong's"):
            st.session_state["strongs_prefix"] = "G"
    
    # Initialize prefix if not set
    if "strongs_prefix" not in st.session_state:
        st.session_state["strongs_prefix"] = "H"
    
    with col3:
        strongs_number = st.sidebar.text_input(
            f"Strong's Number ({st.session_state['strongs_prefix']})",
            placeholder="e.g., 430"
        )
    
    # Combine prefix + number for the final word
    if strongs_number:
        user_word = f"{st.session_state['strongs_prefix']}{strongs_number}"
    else:
        user_word = ""

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

def generate_network(word, depth, similar_count, books):
    """Generate word embedding network without page reload"""

    st.cache_data.clear()
    
    # Handle different input types
    # Check if it's a plain number (old behavior)
    if word.isdigit():
        word = int(word)
    # Check if it's a Strong's number (H#### or G####)
    elif word.startswith(('H', 'G')) and len(word) > 1 and word[1:].isdigit():
        # Keep as string (e.g., "H430", "G2316")
        # VocabNet will handle the Strong's number format
        pass
    # Otherwise it's a Hebrew/Greek word - keep as string
    else:
        pass
    
    
    with st.spinner("Generating network visualization..."):
        if 'network_generated' in st.session_state:
            del st.session_state['network_generated']
            del st.session_state['network_html']
        
        if not user_word:
            st.warning("Please enter a word or number to generate the network.")
        elif 'selected_books' not in st.session_state or not st.session_state['selected_books']:
            st.warning("⚠️ Please select and load Bible books first!")
        else:
            st.info(f"Building network for '{user_word}' with {num_similar} similar words per level and depth of {search_depth}.")
            st.info(f"Using books: {', '.join(st.session_state['selected_books'])}")

            dg = nx.DiGraph()

            NB = NetBuilder()
            
            NB.generate_word_search_network(Algorithm.W2V,
                                        word,
                                        num_steps=depth,
                                        words_per_level=similar_count,
                                        books_to_include=[])
            vnet = NB.get_network()
            # data = nx.node_link_data(vnet)
            net = Network(height="1000px", width="100%", directed=True, bgcolor="#000000", font_color="white")
            
            # Add nodes & edges
            for n, attrs in vnet.nodes(data=True):
                if n == NB.process_strongs_input(word[0], int(word[1:])):
                    net.add_node(n, color='#ffff00', title=n, **attrs)
                else:
                    net.add_node(n, title=n, **attrs)
                
            for u, v, attrs in vnet.edges(data=True):
                print(attrs['weight'])
                net.add_edge(u, v, weight=(attrs['weight']), title=attrs['weight'], label=attrs['weight'], arrows="to")
            
            # # Prep & render
            net.repulsion()
            net.show_buttons()
            # net.show_buttons(filter_=["layout", "interaction", "nodes", "edges"])
            # net.show("sim_graph.html")
            # HtmlFile = open("sim_graph.html", "r", encoding="utf-8")
            html = net.generate_html()
            components.html(html, height=750)
            
            #with st.spinner("Generating network visualization..."):
                #generate_network(user_word, search_depth, num_similar, st.session_state['selected_books'])
            
            st.success("✅ Network generated successfully!")
    
            if 'network_generated' in st.session_state and st.session_state['network_generated']:
                components.html(st.session_state['network_html'], height=800)


# =====================================================
# Bible Book Selector (Multi-select) with Presets
# =====================================================
st.sidebar.markdown("### 📖 Bible Book Selection")
st.sidebar.caption("Select one or more books to analyze")

# Initialize multiselect value if not set
if "multiselect_value" not in st.session_state:
    st.session_state["multiselect_value"] = ["Genesis"]

# --- Preset Buttons ---
        #All books
if st.sidebar.button('Whole Bible'):
    all_books = list(BIBLE_BOOKS.keys())
    st.session_state["selected_books"] = all_books
    st.session_state['multiselect_value'] = all_books

if st.sidebar.button("Whole OT"):
    ot_books = [book for book, sec in BIBLE_BOOKS.items() if sec == "OT"]
    st.session_state["selected_books"] = ot_books
    st.session_state["multiselect_value"] = ot_books

if st.sidebar.button("Whole NT"):
    nt_books = [book for book, sec in BIBLE_BOOKS.items() if sec == "NT"]
    st.session_state["selected_books"] = nt_books
    st.session_state["multiselect_value"] = nt_books

# --- Multiselect (uses session state value) ---
selected_books = st.sidebar.multiselect(
    "Choose Bible Books",
    options=list(BIBLE_BOOKS.keys()),
    default=st.session_state["multiselect_value"],
    help="Select multiple books to train the Word2Vec model",
)

# Keep session state in sync
st.session_state["selected_books"] = selected_books
st.session_state["multiselect_value"] = selected_books

# --- Load Books Button ---
if st.sidebar.button("📥 Load Selected Books", type="primary"):
    if selected_books:
        st.sidebar.success(f"✅ Loaded {len(selected_books)} book(s)")
    else:
        st.sidebar.warning("⚠️ Please select at least one book")
if st.sidebar.button("Generate Word Embedding"):
        # Clear previous network when generating a new one
    generate_network(user_word, search_depth, num_similar, st.session_state['selected_books'])
