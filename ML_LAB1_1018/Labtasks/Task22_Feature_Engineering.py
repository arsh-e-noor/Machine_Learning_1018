import pandas as pd

data = pd.DataFrame({
    "Date": ["2026-01-05", "2026-02-18", "2026-03-15", "2026-04-01"],
    "Height_cm": [158, 168, 178, 188],
    "Weight": [55, 68, 78, 88]
})

data["Date"] = pd.to_datetime(data["Date"])
height_m = data["Height_cm"] / 100
data["BMI"] = data["Weight"] / (height_m ** 2)
data["Day_of_Week"] = data["Date"].dt.day_name()

print(data)
