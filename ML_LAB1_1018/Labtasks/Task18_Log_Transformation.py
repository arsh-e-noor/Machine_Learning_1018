import pandas as pd
import numpy as np

data = pd.DataFrame({"Income": [25000, 27000, 26000, 28000, 500000, 24000]})
data["Income_Log"] = np.log(data["Income"])

print(data)
