from sklearn import datasets

iris = datasets.load_iris()

print("Features:")
print(iris.data[:5])
print("\nTarget values:")
print(iris.target[:5])
print("\nFeature Names:")
print(iris.feature_names)
print("\nTarget Names:")
print(iris.target_names)
print("\nShape:")
print(iris.data.shape)
