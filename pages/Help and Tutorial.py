import streamlit as st

st.set_page_config(
    page_title="GraphLex Tutorial",
    layout="wide"
)

# Sidebar: "Cheat Sheet" 
with st.sidebar:
    st.header("Quick Links")
    st.info("Need a Strong's Number?")
    st.markdown("""
    1. **[BibleHub Interlinear](https://biblehub.com/interlinear/)**
       *(Look for the number above the word)*
    2. **[StepBible](https://www.stepbible.org/)**
       *(Hover over any word and click it)*
    """)
    
    st.divider()
    
    # Feedback button
    st.markdown("Found a bug?")
    st.markdown("[http://example.com]] Click here")
    
    st.divider()
    st.caption("GraphLex v1.0 | Senior Capstone")

#Main Header
st.title("How to Use GraphLex")

# Expander to help keep the top of the page clean
with st.expander("Read Me First: What is this tool?", expanded=True):
    st.markdown("""
    **Welcome to GraphLex!** If you are used to using a standard lexicon, this tool might feel different. 
    * A **Dictionary** tells you what a word *means* (Definitions).
    * **GraphLex** shows you how a word *behaves* (Relationships).

    Think of a dictionary like a **Phone Book**: it simply lists facts. \n 
    GraphLex is like a **Social Network Map**: it shows you who "hangs out" with whom in the biblical text.
    """)

st.markdown("---")

#Tabs 
tab1, tab2, tab3, tab4 = st.tabs([
    "Quick Start", 
    "The Concepts (Visualized)", 
    "Settings Guide", 
    "Interpretation"
])

# Tab 1: Quick Start
with tab1:
    st.header("Get Results in 3 Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. The Input")
        st.markdown("Find your Strong's Number (e.g., `G26` or `H2617`).")
        st.warning("NOTE: Ensure you select 'H' (Hebrew) or 'G' (Greek) in the sidebar toggle!")
        
    with col2:
        st.subheader("2. The Mode")
        st.markdown("Choose your lens:")
        st.markdown("Paradigmatic: Similar meanings.")
        st.markdown("Syntagmatic: Nearby neighbors.")
        
    with col3:
        st.subheader("3. The Graph")
        st.markdown("Click **Generate**. You can drag nodes, zoom in/out, and screenshot the results.")

    st.markdown("---")
    st.markdown("What should it look like?")
    st.info("PICTURE OF OUR APP HERE")

# Tab 2: The Concepts
with tab2:
    st.header("The Logic: Peanut Butter & Jelly")
    st.markdown("Understanding the difference between the two modes is the key to using this tool.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Syntagmatic (Neighbors)")
        st.success("The 'Peanut Butter & Jelly' Relationship")
        st.markdown("""
        * **The Logic:** These words appear *next* to each other.
        * **The Question:** "What words hang out nearby?"
        * **Biblical Example:** *Love* and *Commandments*. They aren't the same thing, but they appear in the same sentence often.
        """)

    with col2:
        st.subheader("2. Paradigmatic (Family)")
        # Changed st.primary to st.warning (Orange) to fix the error
        st.warning("The 'Peanut Butter & Almond Butter' Relationship")
        st.markdown("""
        * **The Logic:** These words are *substitutes* for each other.
        * **The Question:** "What other words work in this context?"
        * **Biblical Example:** *Love* and *Mercy*. They are 'siblings' in meaning.
        """)


    # --- SIMPLIFIED DIAGRAM (No Graphviz required) ---
    st.markdown("---Visualizing the Difference---")
    st.caption("This chart illustrates how the computer analyzes the relationships [INSERT POSSIBLE CHART HERE]:")

    # Create 3 columns for a manual flow chart
    c1, c2, c3 = st.columns([1, 1, 1])

    with c1:
        st.markdown("### Syntagmatic")
        st.markdown("*(Horizontal Context)*")
        st.info("I like **Peanut Butter**...")
        st.markdown("⬇*is followed by*")
        st.success("...and **Jelly**")
        st.caption("They are neighbors.")

    with c2:
        st.markdown("### Paradigmatic")
        st.markdown("*(Vertical Substitution)*")
        st.info("I like **Peanut Butter**...")
        st.markdown("↕*can be swapped with*")
        st.warning("... **Almond Butter**")
        st.caption("They are siblings.")
    
    with c3:
        st.markdown("### Biblical Example")
        st.markdown("*(Applied)*")
        st.write("Target: **Love** (Agapē)")
        st.markdown("-> **Neighbor:** Commandments")
        st.markdown("⬇ **Sibling:** Mercy")

# Tab 3: Settings
with tab3:
    st.header("Tweaking the Engine")
    
    # Columns here help break up the text
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Search Depth")
        st.markdown("""
        Controls how far the net is cast.
        * **Level 1:** Immediate friends only.
        * **Level 2-3:** Friends of friends.
        * **Pro Tip:** Keep this low (1-2) to avoid a messy "spaghetti" graph.
        """)
        
    with c2:
        st.markdown("#### Similarity Count")
        st.markdown("""
        Controls how many connections each word makes.
        * **Higher Number:** Denser, more complex web, but clearer sense of the bigger picture
        * **Lower Number:** Less Clutter, but less of the bigger picture
        """)

# Tab 4: Interpretation
with tab4:
    st.header("From Data to Doctrine")
    st.error("NOTE: GraphLex gives you *data*, not *theology*, a definition, or the actual meaning.")
    
    st.markdown("""
    When you see a connection, don't assume it has a deep spiritual meaning immediately. 
    Use the graph to **generate questions** for your research paper.
    """)
    
    with st.expander("Example: analyzing 'Faith' (Pistis)", expanded=True):
        st.markdown("""
        1. **The Data:** You see a strong Syntagmatic line connecting **Faith** to **Hearing**.
        2. **Bad Interpretation:** "GraphLex says Faith is the same thing as Hearing." (Incorrect)
        3. **Good Interpretation:** "GraphLex shows that Faith and Hearing appear together often. Why? I should look up those verses." (Correct)
        """)
        
    st.markdown("### Troubleshooting")
    with st.expander("My graph is empty!"):
        st.markdown("Try increasing the **Number of Similar Words** or check that you selected a Bible book where that word actually appears.")
        
    with st.expander("My graph is a giant mess!"):
        st.markdown("Lower the **Search Depth** to 1. Deep searches grow exponentially!")

st.markdown("---")
st.caption("GraphLex Project | Created by Rhett Seitz and Rhys Sharpe | Southern Adventist University")