import streamlit as st
import pandas as pd

st.title("About GraphLex")

st.markdown("""
            ## Contributors
This project was a group effort by multiple students and professors, without whom GraphLex would 
            never have been more than an idea.

Licensed as CC-BY-NC 4.0, 2025-2026.
""")

authors = pd.DataFrame([["Rhett Seitz", "B.S. Computer Science", "Application development, writing, applied research", "rhettseitz@southern.edu"],
          ["Rhys Sharpe", "B.A. Archaeology (Near Eastern Studies) and Computer Science", "Research, backend programming, project management", "davidrsharpe@southern.edu"],
          ["Dr. Germán H. Alférez", "Professor, School of Computing", "Project supervisor", "harveya@southern.edu"]])
authors.columns = ["Name", "Major/Position", "Role", "Email"]

st.table(authors)

st.markdown("""



--- 
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