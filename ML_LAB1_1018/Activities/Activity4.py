#23-NTU-CS-1018
#ARSHE NOOR
#BSCS - 7TH A
import pandas as pd
import matplotlib.pyplot as plt


# ------------------------------------------
# 1. IMPORT THE THREE DATASETS
# ------------------------------------------

students = pd.read_csv("../Data sets/students.csv")
attendance = pd.read_json("../Data sets/attendance.json")
extra = pd.read_excel("../Data sets/extra.xlsx")


# Show the loaded datasets
print("STUDENTS DATA:")
print(students)

print("\nATTENDANCE DATA:")
print(attendance)

print("\nEXTRA DATA:")
print(extra)


# ------------------------------------------
# 2. REVIEW MISSING VALUES
# ------------------------------------------

print("\nMissing values in Students:")
print(students.isnull().sum())

print("\nMissing values in Attendance:")
print(attendance.isnull().sum())

print("\nMissing values in Extra:")
print(extra.isnull().sum())


# ------------------------------------------
# 3. PROCESS MISSING VALUES
# ------------------------------------------

students["Marks"] = students["Marks"].fillna(
    students["Marks"].median()
)

attendance["Attendance"] = attendance["Attendance"].fillna(
    attendance["Attendance"].median()
)

extra["Bonus"] = extra["Bonus"].fillna(
    extra["Bonus"].median()
)


# ------------------------------------------
# 4. COMBINE THE DATASETS
# ------------------------------------------

merged_df = pd.merge(
    students,
    attendance,
    on="Name",
    how="inner"
)

merged_df = pd.merge(
    merged_df,
    extra,
    on="Name",
    how="inner"
)


# Organize the columns
merged_df = merged_df[
    ["Name", "Marks", "Attendance", "Bonus"]
]


print("\n===================================")
print("MERGED DATAFRAME")
print("===================================")

print(merged_df)


# ------------------------------------------
# 5. SCATTER GRAPH
# MARKS COMPARED WITH ATTENDANCE
# ------------------------------------------

# Students whose attendance is under 70%
low_attendance = merged_df["Attendance"] < 70


plt.figure(figsize=(10, 6))


# Students whose attendance is 70% or higher
plt.scatter(
    merged_df.loc[~low_attendance, "Attendance"],
    merged_df.loc[~low_attendance, "Marks"],
    label="Attendance >= 70%"
)


# Students whose attendance is below 70%
plt.scatter(
    merged_df.loc[low_attendance, "Attendance"],
    merged_df.loc[low_attendance, "Marks"],
    label="Attendance < 70%"
)


plt.xlabel("Attendance (%)")
plt.ylabel("Marks")
plt.title("Marks vs Attendance")

plt.legend()
plt.grid(True)

plt.show()


# ------------------------------------------
# 6. DISPLAY LOW ATTENDANCE STUDENTS
# ------------------------------------------

print("\n===================================")
print("STUDENTS WITH ATTENDANCE < 70%")
print("===================================")

print(
    merged_df[merged_df["Attendance"] < 70]
)


# ------------------------------------------
# 7. APPLY ONE-HOT ENCODING TO NAME
# ------------------------------------------

encoded_df = pd.get_dummies(
    merged_df,
    columns=["Name"],
    dtype=int
)


print("\n===================================")
print("ONE-HOT ENCODED DATAFRAME")
print("===================================")

print(encoded_df)


# ------------------------------------------
# 8. STORE THE RESULTS
# ------------------------------------------

merged_df.to_csv(
    "../merged_students.csv",
    index=False
)

encoded_df.to_csv(
    "../students_one_hot_encoded.csv",
    index=False
)


print("\nFiles saved successfully!")