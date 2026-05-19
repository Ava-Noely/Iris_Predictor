import streamlit as st 
import numpy as np
from PIL import Image
from sklearn.datasets import load_iris 
from sklearn.ensemble import RandomForestClassifier 
 
st.title("Iris Flower Classifier") 
 
iris = load_iris() 
model = RandomForestClassifier() 
model.fit(iris.data, iris.target) 
 
sepal_length = st.number_input("Sepal Length", min_value=4.0, max_value=8.0, value=5.5, step=0.1)
sepal_width = st.number_input("Sepal Width", min_value=2.0, max_value=4.5, value=3.0, step=0.1)
petal_length = st.number_input("Petal Length", min_value=1.0, max_value=7.0, value=4.0, step=0.1)
petal_width = st.number_input("Petal Width", min_value=0.1, max_value=2.5, value=1.3, step=0.1)
 
if st.button("Predict"): 
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]]) 
    pred = model.predict(features)[0] 
    st.success(f"Prediction: {iris.target_names[pred]}")
    img_path = f"static/imgs/iris_{iris.target_names[pred]}.jpg"
    img = Image.open(img_path)
    st.image(img, caption=iris.target_names[pred], width=300)
