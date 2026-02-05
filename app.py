import streamlit as st
import pandas as pd
import numpy as np
import pickle
import requests
import matplotlib.pyplot as plt

# ---------- PAGE CONFIG ----------
st.set_page_config(page_title="Weather AI", layout="wide")

st.title("🌦 AI/ML Weather Prediction System")

# ---------- UI SIZE CONTROL ----------
st.markdown("""
<style>

.block-container{
    padding-top: 2rem;
    padding-bottom: 1rem;
    max-width: 1100px;
}

h1{
    font-size:28px !important;
}

h2{
    font-size:22px !important;
}

h3{
    font-size:18px !important;
}

label{
    font-size:14px !important;
}

.stButton>button{
    height:35px;
    font-size:14px;
}

</style>
""", unsafe_allow_html=True)


# ---------- LOAD MODEL ----------
model = pickle.load(open("model.pkl","rb"))
le = pickle.load(open("label.pkl","rb"))

# ---------- SHOW ACCURACY ----------
try:
    with open("accuracy.txt") as f:
        acc = f.read()
    st.subheader(f"Model Accuracy: {acc}%")
except:
    pass

st.divider()

# ---------- SPLIT SCREEN ----------
col1, col2 = st.columns(2)

# ================= LEFT SIDE MANUAL =================
with col1:
    st.header("📝 Manual Weather Prediction")

    date = st.date_input("Select Date")
    precipitation = st.number_input("Precipitation",0.0,100.0)
    high_temp = st.number_input("High Temperature",0.0,60.0)
    low_temp = st.number_input("Low Temperature",-10.0,50.0)
    wind = st.number_input("Wind Speed",0.0,50.0)

    if st.button("Predict Manual Weather"):
        df_new = pd.DataFrame([[date,precipitation,high_temp,low_temp,wind]],
        columns=["date","precipitation","temp_max","temp_min","wind"])

        df_new["precipitation"] = np.sqrt(df_new["precipitation"])
        df_new["wind"] = np.sqrt(df_new["wind"])
        df_new["date"] = pd.to_datetime(df_new["date"])
        df_new["date"] = df_new["date"].apply(lambda x: int(x.timestamp()))

        pred = model.predict(df_new)
        result = le.inverse_transform(pred)

        st.success(f"🌤 Predicted Weather: {result[0]}")

# ================= RIGHT SIDE LIVE API =================
with col2:
    st.header("🌍 Live City Weather (Real Time)")

    city = st.text_input("Enter City Name")

    API_KEY = "9f2e4e99d0116e8723716b3d18dbd4c5"

    if st.button("Get Live Weather"):
        if city:
            url = f"https://api.openweathermap.org/data/2.5/weather?q={city}&appid={API_KEY}&units=metric"

            response = requests.get(url)
            data = response.json()

            if data["cod"] == 200:
                temp = data["main"]["temp"]
                wind_speed = data["wind"]["speed"]
                humidity = data["main"]["humidity"]
                rain = data.get("rain", {}).get("1h", 0)

                st.info(f"🌡 Temp: {temp}°C")
                st.info(f"💧 Rain: {rain}")
                st.info(f"💨 Wind: {wind_speed}")

                # convert for model
                df_live = pd.DataFrame([[
                    pd.Timestamp.now(),
                    rain,
                    temp+2,
                    temp-2,
                    wind_speed
                ]], columns=["date","precipitation","temp_max","temp_min","wind"])

                df_live["precipitation"] = np.sqrt(df_live["precipitation"])
                df_live["wind"] = np.sqrt(df_live["wind"])
                df_live["date"] = pd.to_datetime(df_live["date"])
                df_live["date"] = df_live["date"].apply(lambda x: int(x.timestamp()))

                real_weather = data["weather"][0]["main"]

                st.success(f"🌍 Live Weather: {real_weather}")


            else:
                st.error("City not found")

# ---------- GRAPH ----------
st.divider()
st.subheader("📊 Temperature Analysis")

df = pd.read_csv("dataset.csv")
fig, ax = plt.subplots(figsize=(5,3))
ax.scatter(df["temp_max"], df["temp_min"])
ax.set_xlabel("Max Temp")
ax.set_ylabel("Min Temp")
st.pyplot(fig)


