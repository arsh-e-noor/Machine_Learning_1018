import pandas as pd
from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame({
    "Age": [22, 25, 47, 35, 60],
    "Income": [25000, 32000, 95000, 60000, 150000]
})

minmax_scaler = MinMaxScaler()
data_minmax = minmax_scaler.fit_transform(data)

print(data_minmax)
