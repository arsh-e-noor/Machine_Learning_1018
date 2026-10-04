#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
import pandas as pd
import matplotlib.pyplot as plt


# --------------------
# STEP 1: IMPORT THE ONLINE DATASET
# --------------------

url = "https://raw.githubusercontent.com/datasets/covid-19/master/data/time-series-19-covid-combined.csv"

df = pd.read_csv(url)

print("\n================ ORIGINAL DATASET ================\n")
print(df.head())

print("\nDataset shape:")
print(df.shape)

print("\nColumn names:")
print(df.columns.tolist())


# --------------------
# STEP 2: SELECT PAKISTAN RECORDS
# --------------------

pakistan = df[df["Country/Region"] == "Pakistan"].copy()

print("\n================ PAKISTAN DATA ================\n")
print(pakistan.head())

print("\nNumber of Pakistan records:")
print(len(pakistan))


# --------------------
# STEP 3: PROCESS MISSING VALUES
# --------------------

print("\n================ MISSING VALUES ================\n")

print(pakistan.isnull().sum())


# Convert Date to the correct date format
pakistan["Date"] = pd.to_datetime(pakistan["Date"])


# Inspect the key columns
print("\nMissing values in Confirmed:")
print(pakistan["Confirmed"].isnull().sum())


# Fill missing Confirmed values by carrying the previous value forward
pakistan["Confirmed"] = pakistan["Confirmed"].ffill()

# Replace any missing values at the start with 0
pakistan["Confirmed"] = pakistan["Confirmed"].fillna(0)


# --------------------
# STEP 4: ORDER THE DATA BY DATE
# --------------------

pakistan = pakistan.sort_values("Date")


# --------------------
# STEP 5: CALCULATE NEW CASES PER DAY
# --------------------

pakistan["Daily Cases"] = pakistan["Confirmed"].diff()

# Calculate daily cases for the first day
pakistan["Daily Cases"] = pakistan["Daily Cases"].fillna(
    pakistan["Confirmed"]
)

# Avoid negative daily counts caused by reporting corrections
pakistan["Daily Cases"] = pakistan["Daily Cases"].clip(lower=0)


print("\n================ DAILY CASES ================\n")

print(
    pakistan[
        ["Date", "Confirmed", "Daily Cases"]
    ].head(10)
)


# --------------------
# STEP 6: LINE GRAPH - CONFIRMED CASES OVER TIME
# --------------------

plt.figure(figsize=(12, 6))

plt.plot(
    pakistan["Date"],
    pakistan["Confirmed"],
    linewidth=2
)

plt.title("COVID-19 Confirmed Cases in Pakistan Over Time")

plt.xlabel("Date")

plt.ylabel("Confirmed Cases")

plt.xticks(rotation=45)

plt.grid(True, alpha=0.3)

plt.tight_layout()

plt.show()


# --------------------
# STEP 7: FIND THE DAY WITH THE HIGHEST CONFIRMED CASE COUNT
# --------------------

highest_confirmed_index = pakistan["Confirmed"].idxmax()

highest_confirmed_day = pakistan.loc[
    highest_confirmed_index
]

print("\n================ HIGHEST CONFIRMED CASES ================\n")

print(
    "Date:",
    highest_confirmed_day["Date"].date()
)

print(
    "Confirmed Cases:",
    int(highest_confirmed_day["Confirmed"])
)


# --------------------
# STEP 8: CREATE A BOXPLOT OF DAILY CASES
# --------------------

plt.figure(figsize=(8, 6))

plt.boxplot(
    pakistan["Daily Cases"]
)

plt.title("Boxplot of Daily COVID-19 Cases in Pakistan")

plt.ylabel("Daily Cases")

plt.grid(axis="y", alpha=0.3)

plt.show()


# --------------------
# STEP 9: FIND DAILY CASE OUTLIERS WITH IQR
# --------------------

Q1 = pakistan["Daily Cases"].quantile(0.25)

Q3 = pakistan["Daily Cases"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


outliers = pakistan[
    (pakistan["Daily Cases"] < lower_bound) |
    (pakistan["Daily Cases"] > upper_bound)
]


print("\n================ OUTLIER DETECTION ================\n")

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Bound:", lower_bound)

print("Upper Bound:", upper_bound)

print("\nNumber of outlier days:", len(outliers))


if len(outliers) > 0:

    print("\nOutlier days:")

    print(
        outliers[
            ["Date", "Daily Cases"]
        ].to_string(index=False)
    )

else:

    print("\nNo outliers detected.")


# --------------------
# STEP 10: DISPLAY THE FINAL SUMMARY
# --------------------

print("\n================ FINAL SUMMARY ================\n")

print("Country: Pakistan")

print("Total records:", len(pakistan))

print(
    "Highest confirmed cases:",
    int(pakistan["Confirmed"].max())
)

print(
    "Date of highest confirmed cases:",
    pakistan.loc[
        pakistan["Confirmed"].idxmax(),
        "Date"
    ].date()
)

print(
    "Highest daily cases:",
    int(pakistan["Daily Cases"].max())
)

print(
    "Number of daily-case outliers:",
    len(outliers)
)