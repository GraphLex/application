import streamlit as st

st.title("About GraphLex")

st.markdown("""
            ## About the Creators
""")

st.table([["Rhys Sharpe", "B.A. Archaeology (Near Eastern Studies)", "Research, backend programming, project management"],
          ["Rhett Seitz", "B.S. Computer Science", "Application development, writing, applied research"],
          ["Dr. Germán H. Alférez", "Professor, School of Computing", "Project supervisor"]])
            
st.markdown("""


Developed 2025-2026. CC-BY-NC 4.0.

Want to reach out to us? Feel free to email us:
- davidrsharpe@southern.edu
- rhettseitz@southern.edu
- harveya@southern.edu
            
## Implementation
            
### Datasets
            
- Biblia Hebraica Stuttgartensia (Amstelodamensis) (Eep Talstra Center for Bible and Computer, VU University Amsterdam
- OpenHebrewBible (Eliran Wong; modified by Rhys Sharpe)
- Nestle 1904 (Center for Biblical Languages and Computing, Andrews University.

### Source Code
Source code is available at https://github.com/graphlex/application.

## Publications
Coming soon...

""")


st.divider()
st.caption("GraphLex | Rhett Seitz, Rhys Sharpe, and Dr. Germán H. Alférez | CC-BY-NC 4.0 | 2025-2026")