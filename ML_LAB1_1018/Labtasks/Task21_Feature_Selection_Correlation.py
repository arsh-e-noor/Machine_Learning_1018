import pandas as pd

data = pd.DataFrame({
    "Height_cm": [150, 160, 170, 180, 190],
    "Height_inch": [59, 63, 67, 71, 75],
    "Weight": [50, 60, 70, 80, 90]
})

correlation = data.corr()
print(correlation)

data = data.drop("Height_inch", axis=1)
print(data)
