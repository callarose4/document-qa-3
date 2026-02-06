import streamlit as st
lab1_page = st.Page("Labs/Lab1.py", title="Lab 1")
lab2_page = st.Page("Labs/Lab2.py", title="Lab 2")
lab3_page = st.Page("Labs/Lab3.py", title="Lab 3", default=True) #default to Lab 3)


pg = st.navigation([lab1_page, lab2_page, lab3_page])
pg.run()