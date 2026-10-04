#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

# --------------------
# STEP 1: BUILD THE DATASET
# --------------------

data = {
    "Province": [
        "Punjab",
        "Sindh",
        "Khyber Pakhtunkhwa",
        "Balochistan"
    ],

    "Population (millions)": [
        127.69,
        55.70,
        40.86,
        14.89
    ],

    "Literacy Rate (%)": [
        66.25,
        57.54,
        51.09,
        42.01
    ],

    "Region": [
        "East",
        "South",
        "North",
        "West"
    ]
}

df = pd.DataFrame(data)

print("\n================ ORIGINAL DATASET ================\n")
print(df)


# --------------------
# STEP 2: INSERT MISSING VALUES FOR DEMO PURPOSES
# --------------------

# Set Sindh's literacy rate to a missing value
df.loc[df["Province"] == "Sindh", "Literacy Rate (%)"] = np.nan

# Set Balochistan's population to a missing value
df.loc[df["Province"] == "Balochistan", "Population (millions)"] = np.nan

print("\n================ DATASET WITH MISSING VALUES ================\n")
print(df)

print("\nMissing values:")
print(df.isnull().sum())


# --------------------
# STEP 3: REMOVE ONE ROW WITH A MISSING VALUE
# --------------------

# Remove the record where Population is missing
df_dropped = df.dropna(
    subset=["Population (millions)"]
).copy()

print("\n================ AFTER DROPPING ONE ROW ================\n")
print(df_dropped)


# --------------------
# STEP 4: FILL ANOTHER MISSING ENTRY USING THE MEDIAN
# --------------------

median_literacy = df_dropped["Literacy Rate (%)"].median()

df_dropped["Literacy Rate (%)"] = df_dropped[
    "Literacy Rate (%)"
].fillna(median_literacy)

print("\n================ AFTER FILLING MISSING VALUE ================\n")
print(df_dropped)

print("\nMedian Literacy Rate used:")
print(median_literacy)


# --------------------
# STEP 5: APPLY LABEL ENCODING
# --------------------

label_encoder = LabelEncoder()

df_dropped["Region_Label"] = label_encoder.fit_transform(
    df_dropped["Region"]
)

print("\n================ LABEL ENCODING ================\n")
print(df_dropped[
    ["Province", "Region", "Region_Label"]
])

print("\nLabel Encoding Mapping:")

for number, region in enumerate(label_encoder.classes_):
    print(region, "=", number)


# --------------------
# STEP 6: APPLY ONE-HOT ENCODING
# --------------------

one_hot_encoder = OneHotEncoder(
    sparse_output=False,
    handle_unknown="ignore"
)

encoded_regions = one_hot_encoder.fit_transform(
    df_dropped[["Region"]]
)

encoded_columns = one_hot_encoder.get_feature_names_out(
    ["Region"]
)

one_hot_df = pd.DataFrame(
    encoded_regions,
    columns=encoded_columns,
    index=df_dropped.index
)

print("\n================ ONE-HOT ENCODING ================\n")
print(one_hot_df)


# --------------------
# STEP 7: JOIN ONE-HOT DATA WITH THE ORIGINAL DATA
# --------------------

final_df = pd.concat(
    [df_dropped, one_hot_df],
    axis=1
)

print("\n================ FINAL DATASET ================\n")
print(final_df)


# --------------------
# STEP 8: IDENTIFY OUTLIERS WITH IQR
# --------------------

Q1 = final_df["Literacy Rate (%)"].quantile(0.25)

Q3 = final_df["Literacy Rate (%)"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = final_df[
    (final_df["Literacy Rate (%)"] < lower_bound) |
    (final_df["Literacy Rate (%)"] > upper_bound)
]

print("\n================ OUTLIER DETECTION ================\n")

print("Q1:", Q1)
print("Q3:", Q3)
print("IQR:", IQR)
print("Lower Bound:", lower_bound)
print("Upper Bound:", upper_bound)

if len(outliers) == 0:
    print("\nNo literacy-rate outlier detected.")
else:
    print("\nLiteracy-rate outlier(s):")
    print(outliers[
        ["Province", "Literacy Rate (%)"]
    ])


# --------------------
# STEP 9: GENERATE A SCATTER PLOT
# --------------------

plt.figure(figsize=(10, 6))

plt.scatter(
    final_df["Population (millions)"],
    final_df["Literacy Rate (%)"],
    s=100
)

# Place province names next to the points
for _, row in final_df.iterrows():

    plt.annotate(
        row["Province"],
        (
            row["Population (millions)"],
            row["Literacy Rate (%)"]
        ),
        xytext=(5, 5),
        textcoords="offset points"
    )


plt.title(
    "Population vs Literacy Rate of Pakistani Provinces"
)

plt.xlabel("Population (millions)")

plt.ylabel("Literacy Rate (%)")

plt.grid(True)

plt.tight_layout()

plt.show()