import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

#Indlæs fil (gem filen med data og filen med kode i samme mappe for at den kan indlæse)
df = pd.read_csv("Sales_with_NaNs_v1.3.csv")
df.columns = df.columns.str.strip()  #Fjern whitespcace i kolonnenavne

#Debug print
print("Columns in the dataset:", df.columns.tolist())

# --- Skab target variabel til classification --- #
df["Satisfaction_Change"] = df["Customer_Satisfaction_After"] - df["Customer_Satisfaction_Before"]
df["Satisfaction_Label"] = df["Satisfaction_Change"].apply(
    lambda x: 1 if x > 0 else (-1 if x < 0 else 0)
)

#Fjerner rækker hvor target er NaN
df = df.dropna(subset=["Satisfaction_Label"])

# --- Udforsk den manglende data --- #
plt.figure(figsize=(12, 6))
sns.heatmap(df.isnull(), cbar=False, cmap="plasma", yticklabels=False)
plt.title("Missing Data Heatmap")
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# Correlation of missingness (Hjælper med at identificere MCAR vs MAR)
missing_map = df.isnull().astype(int)
plt.figure(figsize=(8, 6))
sns.heatmap(missing_map.corr(), annot=True, cmap="coolwarm")
plt.title("Correlation of Missingness (Helps MAR Detection)")
plt.tight_layout()
plt.show()

# --- Klargør til imputering og modellering ---
target = "Satisfaction_Label"
features = df.drop(columns=[target])
labels = df[target]

num_cols = features.select_dtypes(include='number').columns
features_numeric = features[num_cols]
labels = labels.loc[features_numeric.index]

# ---  Mean Imputering ---
print("\n--- Mean Imputation ---")
mean_imputer = SimpleImputer(strategy='mean')
X_mean = mean_imputer.fit_transform(features_numeric)

X_train, X_test, y_train, y_test = train_test_split(X_mean, labels, test_size=0.2, random_state=42)

clf = RandomForestClassifier(random_state=42)
clf.fit(X_train, y_train)
y_pred = clf.predict(X_test)

print("Classification Report (Mean Imputation):")
print(classification_report(y_test, y_pred))
print("Confusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# --- KNN Imputering ---
print("\n--- KNN Imputation ---")
knn_imputer = KNNImputer(n_neighbors=3)
X_knn = knn_imputer.fit_transform(features_numeric)

X_train2, X_test2, y_train2, y_test2 = train_test_split(X_knn, labels, test_size=0.2, random_state=42)

clf2 = RandomForestClassifier(random_state=42)
clf2.fit(X_train2, y_train2)
y_pred2 = clf2.predict(X_test2)

print("Classification Report (KNN Imputation):")
print(classification_report(y_test2, y_pred2))
print("Confusion Matrix:")
print(confusion_matrix(y_test2, y_pred2))

# --- Fjern Missing Data (Baseline) ---
print("\n--- Drop Missing Data ---")
df_dropna = df.dropna()
if not df_dropna.empty:
    X_drop = df_dropna[num_cols]
    y_drop = df_dropna[target]
    
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X_drop, y_drop, test_size=0.2, random_state=42)

    clf3 = RandomForestClassifier(random_state=42)
    clf3.fit(X_train3, y_train3)
    y_pred3 = clf3.predict(X_test3)

    print("Classification Report (Drop Missing):")
    print(classification_report(y_test3, y_pred3))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test3, y_pred3))
else:
    print("Too little data after dropping NaNs — skipping this part.")

# ---  Dan Relational Database (SQLite) ---
import sqlite3

print("\n--- Creating SQLite Relational Database ---")

#Skaber unikt kunde ID for hver kunde#
df['Customer_ID'] = range(1, len(df) + 1)

# Definer relationalle tabeller
customers_df = df[['Customer_ID', 'Group', 'Customer_Segment']]
sales_records_df = df[['Customer_ID',
                       'Sales_Before', 'Sales_After',
                       'Customer_Satisfaction_Before', 'Customer_Satisfaction_After',
                       'Purchase_Made']]

# Gemmer i SQLite
db_path = "customer_sales.db"
conn = sqlite3.connect(db_path)

customers_df.to_sql("Customers", conn, if_exists="replace", index=False)
sales_records_df.to_sql("Sales_Records", conn, if_exists="replace", index=False)

# Tjek og print sample
cursor = conn.cursor()
cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
print("Tables in database:", [t[0] for t in cursor.fetchall()])

print("\nSample rows from Customers:")
for row in cursor.execute("SELECT * FROM Customers LIMIT 5;"):
    print(row)

print("\nSample rows from Sales_Records:")
for row in cursor.execute("SELECT * FROM Sales_Records LIMIT 5;"):
    print(row)

conn.close()
print(f"\nSQLite database saved as '{db_path}'")
