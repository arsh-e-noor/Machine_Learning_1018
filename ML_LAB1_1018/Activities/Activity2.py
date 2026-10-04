#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# --------------------
# STEP 1: CREATE RANDOM DATA FOR 100 STUDENTS
# --------------------

np.random.seed(42)

students = 100

data = {
    "ID": range(1, students + 1),

    "Math": np.random.randint(40, 101, students),

    "Science": np.random.randint(40, 101, students),

    "English": np.random.randint(40, 101, students)
}

df = pd.DataFrame(data)


# --------------------
# STEP 2: BUILD THE GRADE COLUMN
# --------------------

def assign_grade(average):
    if average >= 80:
        return "A"
    elif average >= 60:
        return "B"
    else:
        return "C"


df["Average"] = df[["Math", "Science", "English"]].mean(axis=1)

df["Grade"] = df["Average"].apply(assign_grade)

# Temporarily remove Average
df.drop(columns=["Average"], inplace=True)


print("\n================ ORIGINAL DATA ================\n")
print(df.head(10))


# --------------------
# STEP 3: INSERT SOME MISSING MARKS
# --------------------

df.loc[5, "Math"] = np.nan
df.loc[15, "Science"] = np.nan
df.loc[25, "English"] = np.nan
df.loc[35, "Math"] = np.nan
df.loc[45, "Science"] = np.nan

print("\n================ MISSING VALUES ================\n")

print(df.isnull().sum())


# --------------------
# STEP 4: REPLACE MISSING MARKS WITH THE MEAN
# --------------------

subjects = ["Math", "Science", "English"]

for subject in subjects:

    mean_value = df[subject].mean()

    df[subject] = df[subject].fillna(mean_value)


print("\n================ AFTER FILLING MISSING VALUES ================\n")

print(df.isnull().sum())


# --------------------
# STEP 5: ENCODE THE GRADE COLUMN
# --------------------

grade_mapping = {
    "A": 0,
    "B": 1,
    "C": 2
}

df["Grade_Encoded"] = df["Grade"].map(grade_mapping)


print("\n================ GRADE ENCODING ================\n")

print(
    df[["ID", "Grade", "Grade_Encoded"]].head(10)
)


# --------------------
# STEP 6: COMPUTE THE TOTAL SCORE
# --------------------

df["Total Score"] = (
    df["Math"] +
    df["Science"] +
    df["English"]
)


print("\n================ TOTAL SCORES ================\n")

print(
    df[["ID", "Math", "Science", "English", "Total Score"]].head(10)
)


# --------------------
# STEP 7: CREATE A HISTOGRAM OF MATH SCORES
# --------------------

plt.figure(figsize=(8, 5))

plt.hist(
    df["Math"],
    bins=10,
    edgecolor="black"
)

plt.title("Distribution of Math Scores")

plt.xlabel("Math Score")

plt.ylabel("Number of Students")

plt.grid(axis="y", alpha=0.3)

plt.show()


# --------------------
# STEP 8: CREATE A BOXPLOT OF TOTAL SCORES
# --------------------

plt.figure(figsize=(8, 5))

plt.boxplot(
    df["Total Score"],
    vert=True
)

plt.title("Boxplot of Total Student Scores")

plt.ylabel("Total Score")

plt.grid(axis="y", alpha=0.3)

plt.show()


# --------------------
# STEP 9: DETERMINE OUTLIERS
# USING THE IQR APPROACH
# --------------------

Q1 = df["Total Score"].quantile(0.25)

Q3 = df["Total Score"].quantile(0.75)

IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR

upper_bound = Q3 + 1.5 * IQR


outliers = df[
    (df["Total Score"] < lower_bound) |
    (df["Total Score"] > upper_bound)
]


print("\n================ OUTLIER DETECTION ================\n")

print("Q1:", Q1)

print("Q3:", Q3)

print("IQR:", IQR)

print("Lower Bound:", lower_bound)

print("Upper Bound:", upper_bound)


if len(outliers) == 0:

    print("\nNo unusually high or low total scores found.")

else:

    print("\nUnusually high/low students:")

    print(
        outliers[
            ["ID", "Math", "Science", "English", "Total Score"]
        ]
    )


# --------------------
# STEP 10: FIND THE HIGHEST-PERFORMING GRADE
# --------------------

df["Average Score"] = (
    df["Math"] +
    df["Science"] +
    df["English"]
) / 3


grade_performance = df.groupby("Grade")["Average Score"].mean()


print("\n================ GRADE PERFORMANCE ================\n")

print(grade_performance)


best_grade = grade_performance.idxmax()

best_score = grade_performance.max()


print(
    f"\nGrade {best_grade} performed best overall "
    f"with an average score of {best_score:.2f}"
)


# --------------------
# STEP 11: DISPLAY THE FINAL DATASET
# --------------------

print("\n================ FINAL DATASET ================\n")

print(df.head(10))

print("\nTotal number of students:", len(df))