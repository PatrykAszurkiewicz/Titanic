import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import Colormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.impute import KNNImputer
from sklearn.impute import SimpleImputer

titanic = pd.read_csv('train.csv')

print(titanic)
plt.figure(figsize=(10,5))
print(titanic.describe())

titanic["Sex"] = titanic["Sex"].map({"male": 0, "female": 1})
titanic["Embarked"] = titanic["Embarked"].map({"C": 0, "Q": 1, "S": 2})

#Wypełnianie KNNImputer oraz SimpleImputer
features = ["Age", "Pclass", "Fare", "SibSp", "Parch"]
age_data = titanic[features]
imputer = KNNImputer(n_neighbors=5)
titanic["Age"] = imputer.fit_transform(age_data)[:, 0]

imputer = SimpleImputer(strategy="most_frequent")
titanic["Embarked"] = imputer.fit_transform(titanic[["Embarked"]])

#Wypełnianie ręczne
#titanic["Embarked"].fillna(2, inplace=True)
#titanic["Age"].fillna(titanic["Age"].median(), inplace=True)

#usunięcie niepotrzebnych danych
titanic = titanic.drop(columns=["Name", "Ticket", "Cabin"])

corr_matrix = titanic.select_dtypes(include=['number']).corr()
#Tworzenie wykresu heatmap
sns.heatmap(corr_matrix, cmap="YlGnBu", annot=True, linewidths= 0.75)
plt.title("Mapa cieplna korelacji zmiennych w zbiorze Titanic")
plt.show()
#Podział na część nauki i testową
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
#Wykres ile przeżyło osób
sns.catplot(x="Pclass", hue="Sex", col="Survived", data=titanic[titanic["Survived"] == 1], kind="count", palette="coolwarm")
plt.subplots_adjust(top=0.85)
plt.suptitle("Przeżycie klas i płeć")
plt.show()
#Sprawdzenie brakujących danych
sns.heatmap(titanic.isnull(), cmap="viridis", cbar=False, yticklabels=False)
plt.title("Mapa brakujących danych w zbiorze Titanic")
plt.show()
#Wypisanie ile czego brakuje
missing_percentage = (titanic.isnull().sum() / len(titanic)) * 100
print(missing_percentage)

#RandomForest
model = RandomForestClassifier(n_estimators=500, random_state=42)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"Dokładność modelu: {accuracy_score(y_test, y_pred):.2f}")
print("Raport RandomForest:\n", classification_report(y_test, y_pred))

sns.countplot(x=y_pred, hue=y_pred, palette="coolwarm")
plt.title("Przewidywane wyniki (0 = nie przeżył, 1 = przeżył)")
plt.xlabel("Survived")
plt.ylabel("Liczba osób")
plt.show()

#LogisticRegression \\ saga < liblinear
model = LogisticRegression(max_iter=25000, solver="liblinear")
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
print(f"Dokładność modelu: {accuracy:.2f}")
print("Raport LogisticRegression:\n", classification_report(y_test, y_pred))