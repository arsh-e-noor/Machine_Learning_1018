#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------
# 1. BUILD THE DATASET
# ------------------------------------------

data = {
    "Name": ["Ahsan", "Hira", "Bilal", "Zara", "Salman", "Mahnoor"],
    "Age": [25, 27, 35, 29, None, 40],
    "Salary": [50000, None, 75000, 2000000, 60000, 90000],
    "Department": ["IT", "Finance", "IT", "HR", "Finance", "IT"]
}

df = pd.DataFrame(data)

print("Original Data:")
print(df)


# ------------------------------------------
# 2. REVIEW MISSING VALUES
# ------------------------------------------

print("\nMissing Values:")
print(df.isnull().sum())


# ------------------------------------------
# 3. PROCESS MISSING VALUES
# ------------------------------------------

# Replace missing Age values with the median
df["Age"] = df["Age"].fillna(df["Age"].median())

# Replace missing Salary values with the median
df["Salary"] = df["Salary"].fillna(df["Salary"].median())


print("\nData After Handling Missing Values:")
print(df)


# ------------------------------------------
# 4. ENCODE THE DEPARTMENT COLUMN
# ------------------------------------------

df_encoded = pd.get_dummies(
    df,
    columns=["Department"],
    dtype=int
)

print("\nData After One-Hot Encoding:")
print(df_encoded)


# ------------------------------------------
# 5. CREATE A SALARY BOXPLOT
# ------------------------------------------

plt.figure(figsize=(8, 5))

plt.boxplot(df["Salary"])

plt.title("Salary Outlier Detection")
plt.ylabel("Salary")

plt.grid(True)

plt.show()


# ------------------------------------------
# 6. IDENTIFY OUTLIERS WITH IQR
# ------------------------------------------

Q1 = df["Salary"].quantile(0.25)
Q3 = df["Salary"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

print("\nQ1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)

print("\nLower Bound:", lower_bound)
print("Upper Bound:", upper_bound)


# Identify the outlier records
outliers = df[
    (df["Salary"] < lower_bound) |
    (df["Salary"] > upper_bound)
]

print("\nSalary Outliers:")
print(outliers[["Name", "Salary"]])


# ------------------------------------------
# 7. REVIEW ZARA'S SALARY
# ------------------------------------------

zara_salary = df.loc[df["Name"] == "Zara", "Salary"].iloc[0]

print("\nZara's Salary:", zara_salary)

if zara_salary > upper_bound or zara_salary < lower_bound:
    print("Zara's salary is an outlier.")
else:
    print("Zara's salary is not an outlier.")