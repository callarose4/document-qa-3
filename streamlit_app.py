import streamlit as st
lab1_page = st.Page("Labs/Lab1.py", title="Lab 1")
lab2_page = st.Page("Labs/Lab2.py", title="Lab 2")
lab3_page = st.Page("Labs/Lab3.py", title="Lab 3")
lab4_page = st.Page("Labs/Lab4.py", title="Lab 4")
lab5_page = st.Page("Labs/Lab5.py", title="Lab 5", default=True)


pg = st.navigation([lab1_page, lab2_page, lab3_page, lab4_page, lab5_page])
pg.run()