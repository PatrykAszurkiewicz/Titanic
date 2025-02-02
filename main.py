import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap
from sklearn.model_selection import train_test_split

titanic = pd.read_csv('train.csv')

print(titanic)
plt.figure(figsize=(10,5))
print(titanic.describe())

titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
titanic["Embarked"] = titanic["Embarked"].map({"C": 0, "Q": 1, "S": 2})
titanic["Embarked"].fillna(2, inplace=True)

titanic["Age"].fillna(titanic["Age"].median(), inplace=True)

titanic = titanic.drop(columns=["Name", "Ticket", "Cabin"])

corr_matrix = titanic.select_dtypes(include=['number']).corr()

sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, linewidths= 0.75)
plt.title("Mapa cieplna korelacji zmiennych w zbiorze Titanic")
plt.show()

X = titanic.drop(columns=["Survived"])
y = titanic["Survived"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

plt.subplot(1, 2, 1)
sns.countplot(x = y_train, hue=y_train, palette="coolwarm")
plt.title("Trening")

plt.subplot(1, 2, 2)
sns.countplot(x = y_test, hue=y_test, palette="coolwarm")
plt.title("Test")

plt.show()

sns.catplot(x="Pclass", hue="Sex", col="Survived", data=titanic[titanic["Survived"] == 1], kind="count", palette="coolwarm")
plt.subplots_adjust(top=0.85)
plt.suptitle("Przeżycie klas i płeć")
plt.show()

sns.heatmap(titanic.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Mapa brakujących danych w zbiorze Titanic")
plt.show()

missing_percentage = (titanic.isnull().sum() / len(titanic)) * 100
print(missing_percentage)

