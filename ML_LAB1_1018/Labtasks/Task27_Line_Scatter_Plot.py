import matplotlib.pyplot as plt

year = [1994, 1995, 1998, 2000]
population = [2.59, 3.69, 5.33, 6.77]

plt.plot(year, population)
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Over Time")
plt.show()

plt.scatter(year, population)
plt.xlabel("Year")
plt.ylabel("Population")
plt.title("Population Scatter Plot")
plt.show()
