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
    st.markdown("Found a bug? [Report it here](https://wkf.ms/4rBJEOl)")
    st.divider()
    # st.caption("GraphLex v1.0 | Senior Capstone")

    st.header("Glossary")
    st.markdown("""
                - *graph* - In computer science, a graph is a data structure containing nodes and edges that connect those nodes.
                - *network* - Another computer science term, very similar in meaning to 'graph.'
                - *lemma*
                - *lexeme*
                - *syntagmatic*
                - *paradigmatic*
                """)

#Main Header
st.title("How to Use GraphLex")

# Expander to help keep the top of the page clean
with st.expander("Read Me First: What is this tool?", expanded=True):
    st.markdown("""
    **Welcome to GraphLex!**

    GraphLex is a *semantic search engine* designed for lexical analysis (word studies). It's designed with students, researchers, and ministers in mind (though it's open to anyone interested in learning more about Hebrew and Greek semantics).
                
    If you are used to using a standard lexicon, this tool might feel different. 
    * A **dictionary or lexicon** tells you what a word *means* (definitions).
    * **GraphLex** shows you how a word *behaves* (relationships).

    Think of a dictionary like a **Phone Book**: it simply lists facts. \n 
    GraphLex is like a **Social Network Map**: it shows you who "hangs out" with whom in the biblical text.
    """)

    
    st.error('NOTE: GraphLex gives you *data*, not *theology*, a definition, or the actual meaning. See "Interpretation" below for more information.')


st.markdown("---")

#Tabs 
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "Quick Start",
    "The Concepts",
    "Settings Guide", 
    "Interpretation",
    "Troubleshooting/FAQ",
    "Limitations/Bugs"
])

# Tab 1: Quick Start
with tab1:
    st.header("Get Results in 3 Steps")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.subheader("1. The Word")
        st.markdown("Find the Strong's Number (e.g., `G26` or `H2617`) for your chosen word. Enter the number into the Strong's Number field in the sidebar.")
        st.warning("NOTE: Ensure you select 'H' (Hebrew) or 'G' (Greek) in the sidebar!")
        
    with col2:
        st.subheader("2. The Mode")
        st.markdown("Choose your lens:")
        st.markdown("*Paradigmatic:* Paradigmatic relationships are those based on semantic similarity as measured by the Word2Vec algorithm. For instance, 'cat' and 'dog' would have high correlation.")
        st.markdown("*Syntagmatic:* Syntagmatic relationships are those based on co-occurrence counts. For instance, 'dog' and 'barks' would have high correlation.")
        
    with col3:
        st.subheader("3. The Graph")
        st.markdown("""Click **Generate**. You can drag nodes, zoom in/out, and screenshot the results. \n\n Click a node to see details--the Strong's number, an English gloss, etc. The yellow node is the term you entered.""")

    # st.markdown("---")
    # st.markdown("What should it look like?")
    # st.info("PICTURE OF OUR APP HERE")

# Tab 2: The Concepts
with tab2:
    st.header("The Logic: Peanut Butter & Jelly")
    st.markdown("Understanding the difference between the two modes is key to using this tool.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader("1. Syntagmatic")
        st.success("The 'Peanut Butter & Jelly' Relationship")
        st.markdown("""
        * **The Logic:** These words appear *next* to each other.
        * **The Question:** "What words co-occur (appear near each other)?"
        * **Biblical Example:** *Love* and *Commandments*. They aren't the same thing, but they appear in the same sentence often.
        """)
# TODO: Could you use a Greek example for this?

    with col2:
        st.subheader("2. Paradigmatic")
        # Changed st.primary to st.warning (Orange) to fix the error
        st.warning("The 'Peanut Butter & Almond Butter' Relationship")
        st.markdown("""
        * **The Logic:** These words are *substitutes* for each other.
        * **The Question:** "What other words work in this context?"
        * **Biblical Example:** *Love* and *Mercy*. They are 'siblings' in meaning.
        """)


    # --- SIMPLIFIED DIAGRAM (No Graphviz required) ---
    # st.markdown("---Visualizing the Difference---")
    # st.caption("This chart illustrates how the computer analyzes the relationships [INSERT POSSIBLE CHART HERE]:")

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

# TODO: I don't think I'm following the layout here...

# Tab 3: Settings
with tab3:
    st.header("Tweaking the Engine")
    
    # Columns here help break up the text
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("#### Search Depth")
        st.markdown("""
        Controls how many levels GraphLex will search.
                                      
        GraphLex will search for a certain number of connections, then search for connections for each of *those* connections in turn. Search Depth controls how many steps that runs for.
        * **Level 1:** Immediate friends only.
        * **Level 2-3:** Friends of friends.
        * **Pro Tip:** Keep this low (1-3) to avoid a messy "spaghetti" graph.
        """)
        
    with c2:
        st.markdown("#### Similar Words per Level")
        st.markdown("""
        Controls how many connections are shown for each word.
                    
        Setting Similar Words per Level to 10 and Search Depth to 2 will yield the 10 strongest connections for your chosen word, then the 10 strongest connections for each of those 10 connected words.

        * **Higher Number:** Denser, more complex web, but clearer sense of the bigger picture
        * **Lower Number:** Less clutter, but less of the bigger picture
        """)

# Tab 4: Interpretation
with tab4:
    st.header("From Data to Doctrine")
    
    st.markdown("""
    When you see a connection, don't assume it has a deep spiritual meaning immediately. 
    Use the graph to generate *questions* rather than *answers.*
    """)


    st.info("**Exegesis Tips**:\n\nBe aware of polysemy - many words can have more than one meaning, which these datasets may not not distinguish.\n\n"
                "Do not assume theological significance to a relation simply because it's there. Ask why--why are these words related?"
    )
    
    with st.expander("Example: analyzing 'Faith'", expanded=True):
        st.markdown("""
        1. **The Data:** You see a strong Syntagmatic line connecting **Faith** to **Hearing**.
        2. **Bad Interpretation:** "GraphLex says Faith is the same thing as Hearing." 
        3. **Good Interpretation:** "GraphLex shows that Faith and Hearing appear together often. Why? I should look up those verses."
        """)
        
with tab5:
    st.header("Troubleshooting / FAQ")
    with st.expander("My graph is empty!"):
        st.markdown("Try increasing the **Number of Similar Words** or check that you selected a Bible book where that word actually appears.")
        
    with st.expander("My graph is a giant mess!"):
        st.markdown("Lower the **Search Depth** to 1. Deep searches grow exponentially!")

    with st.expander("I don't understand what appears in the Strong's number field!"):
        st.markdown("""
                    - Sometimes multiple Strong's numbers will appear. Check them all out--this is an effect of the specific dataset used.
                    - If you see E0, that's an error code--the Strong's number does not appear in the filtered dataset despite being in the paradigmatic model. This is a bug that needs to be worked out, probably by retraining.
                    - If you still don't understand, reach out to us or file a bug report--it could be an error somewhere in the program.
                    """)

with tab6:
    st.header("What GraphLex Doesn't Do (Yet!)")
    st.markdown("""
        - **Ezra and Daniel are not included**. Trying to deal with Aramaic is a future goal.
        - **Words that occur less than 5 times are not included in Paradigmatic.** This is a limitation of the Word2Vec algorithm.
        - **Words such as prepositions and articles (called *stopwords*) are not included and may cause an error if you search for them.** You really don't want to know how often "God" and "the" show up together, do you? If you do, please file a bug report. I want to know about this project.
        
    """)


    
st.divider()
st.caption("GraphLex | Rhett Seitz, Rhys Sharpe, and Dr. Germán H. Alférez | CC-BY-NC 4.0 | 2025-2026")