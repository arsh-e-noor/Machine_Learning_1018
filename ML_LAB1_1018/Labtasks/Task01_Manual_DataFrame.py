#23-ntu-cs-1018
import pandas as pd

data = {
    "Name": ["Alice", "Bob", "Charlie", "Diana"],
    "Age": [25, 30, 35, 28],
    "Salary": [50000, 60000, 70000, 55000]
}
df_manual = pd.DataFrame(data)

print("Manual DataFrame:")
print(df_manual)
