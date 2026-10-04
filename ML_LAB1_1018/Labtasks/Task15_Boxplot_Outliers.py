import pandas as pd
import matplotlib.pyplot as plt

data = pd.DataFrame({"Income": [25000, 27000, 26000, 28000, 500000, 24000]})

plt.boxplot(data["Income"])
plt.title("Income Outlier Detection")
plt.ylabel("Income")
plt.show()
