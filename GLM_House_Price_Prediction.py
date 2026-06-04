import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("train.csv")
print(df.head())

print(df.shape)

print(df.info())

features = [
    "GrLivArea",
    "OverallQual",
    "YearBuilt",
    "TotalBsmtSF"
]

df = df[features + ["SalePrice"]].dropna()
print(df.head())

df["SalePrice"] = np.log(df["SalePrice"])

from sklearn.model_selection import train_test_split

X = df[features]
y = df["SalePrice"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(X_train.shape, X_test.shape)

import statsmodels.api as sm

X_train_sm = sm.add_constant(X_train)

model = sm.GLM(
    y_train,
    X_train_sm,
    family=sm.families.Gaussian()
)

result = model.fit()

print(result.summary())

X_test_sm = sm.add_constant(X_test)
pred = result.predict(X_test_sm)

from sklearn.metrics import mean_absolute_error

mae = mean_absolute_error(y_test, pred)
print(mae)
coefficients = result.params[1:]

coefficients = result.params[1:]

plt.figure(figsize=(8, 5))
coefficients.plot(kind="bar")

plt.title("GLM Coefficients")
plt.xlabel("Features")
plt.ylabel("Coefficient Value")
plt.xticks(rotation=45)
plt.tight_layout()

plt.show()
