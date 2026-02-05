import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

def process_inputs(inputs):

    # ---------------- NEW USER INPUT ----------------
    new_input = [
        inputs[0],        # date
        float(inputs[1]), # precipitation
        float(inputs[2]), # temp_max
        float(inputs[3]), # temp_min
        float(inputs[4])  # wind
    ]

    # ---------------- LOAD DATASET ----------------
    df3 = pd.read_csv("dataset.csv")

    # ---------------- DATE CONVERT ----------------
    df3["date"] = pd.to_datetime(df3["date"])
    df3["date"] = df3["date"].apply(lambda x: int(x.timestamp()))

    # ---------------- LABEL ENCODING ----------------
    lc = LabelEncoder()
    df3["weather"] = lc.fit_transform(df3["weather"])

    # ---------------- HANDLE SKEW ----------------
    df3["precipitation"] = np.sqrt(df3["precipitation"])
    df3["wind"] = np.sqrt(df3["wind"])

    # ---------------- FEATURES & LABEL ----------------
    X = df3.drop("weather", axis=1)
    y = df3["weather"]

    # ---------------- TRAIN TEST SPLIT ----------------
    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # ---------------- BEST MODEL ----------------
    model = RandomForestClassifier(
        n_estimators=500,
        max_depth=25,
        random_state=42
    )
    model.fit(x_train, y_train)

    # ---------------- ACCURACY ----------------
    from sklearn.metrics import accuracy_score
    y_pred = model.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)

    with open("accuracy.txt", "w") as f:
        f.write(str(round(accuracy * 100, 2)))

    # ---------------- SAVE MODEL ----------------
    import pickle
    pickle.dump(model, open("model.pkl","wb"))
    pickle.dump(lc, open("label.pkl","wb"))
    print("MODEL SAVED SUCCESSFULLY")

    # ---------------- NEW INPUT ----------------
    df_new = pd.DataFrame([new_input], columns=[
        "date","precipitation","temp_max","temp_min","wind"
    ])

    # same preprocessing
    df_new["precipitation"] = np.sqrt(df_new["precipitation"])
    df_new["wind"] = np.sqrt(df_new["wind"])

    df_new["date"] = pd.to_datetime(df_new["date"])
    df_new["date"] = df_new["date"].apply(lambda x: int(x.timestamp()))

    # ---------------- PREDICT ----------------
    prediction = model.predict(df_new)
    predicted_weather = lc.inverse_transform(prediction)

    return predicted_weather[0]


# run once to generate model + accuracy
if __name__ == "__main__":
    dummy = ["2012-01-01",0,20,10,5]
    process_inputs(dummy)
