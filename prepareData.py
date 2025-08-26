import pandas as pd

# Load CSV
df = pd.read_csv("api_call_dataset.csv")

print(df.shape)
print(df.columns[:5])  # Check first few column names
print(df["malware"].value_counts())  # Check class balance

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

# Prepare data
X = df.drop(columns=["hash", "malware"])
y = df["malware"]

# Split with stratification
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Train with class weight to handle imbalance
rf = RandomForestClassifier(n_estimators=100, class_weight="balanced", random_state=42)
rf.fit(X_train, y_train)

# Evaluate
y_pred = rf.predict(X_test)
print(classification_report(y_test, y_pred))

