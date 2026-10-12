# ============================================================================
#  Name             : ARSHENOOR
#  Registration No. : 23-NTU-CS-1018
#  Section          : BSCS 7th A
#  Course           : Machine Learning Lab
#  Lab              : Lab 04  -  Voting Classifier
#  Dataset          : Wine dataset (built into sklearn, nothing to download)
# ============================================================================

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score

# ---------------------------------------------------------------------------
# Load and split the data
# ---------------------------------------------------------------------------
wine = load_wine()
X, y = wine.data, wine.target

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42)

# Logistic Regression and SVC need scaled features to behave properly
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# Build the individual classifiers and the hard voting classifier
# ---------------------------------------------------------------------------
log_clf = LogisticRegression()
rnd_clf = RandomForestClassifier()
svm_clf = SVC()

voting_clf = VotingClassifier(
    estimators=[('lr', log_clf), ('rf', rnd_clf), ('svc', svm_clf)],
    voting='hard')

voting_clf.fit(X_train, y_train)

# Predict a single sample (the slide calls this 'some_digit')
some_digit = X_test[0].reshape(1, -1)
print("Prediction for one sample:", voting_clf.predict(some_digit))
print("Actual label             :", y_test[0])
print()

# ---------------------------------------------------------------------------
# Compare every classifier against the voting classifier
# ---------------------------------------------------------------------------
for clf in (log_clf, rnd_clf, svm_clf, voting_clf):
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    print(clf.__class__.__name__, accuracy_score(y_test, y_pred))