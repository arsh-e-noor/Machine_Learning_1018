import pandas as pd

data = pd.DataFrame({
    "Education": ["High School", "Bachelors", "Masters", "PhD", "Bachelors"]
})
order = {"High School": 0, "Bachelors": 1, "Masters": 2, "PhD": 3}

data["Education_Encoded"] = data["Education"].map(order)

print(data)
