import streamlit as st
import pickle
import numpy as np
import pandas as pd

model = pickle.load(open("knn.pkl", "rb"))

st.title("Football Team Rating Prediction")
g = st.number_input("Goals")
s = st.number_input("Shots pg")
y = st.number_input("Yellow Cards")
r = st.number_input("Red Cards")
po = st.number_input("Possession %")
pa = st.number_input("Pass %")
ae = st.number_input("Aerials Won")

if st.button("Predict"):
	test = np.array([[g, s, y, r, po, pa, ae]])
	res = model.predict(test)
	print(res)
	st.success("Rating: " + str(res[0]))
