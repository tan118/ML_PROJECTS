import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import joblib
from sklearn.metrics import accuracy_score

# Load the dataset
df=pd.read_csv('Impact_of_Remote_Work_on_Mental_Health.csv')
print("Initial shape:", df.shape)
print(df.head())
print(df.columns)


# CHECK UNIQUE VALUES (DEBUG)

print("Sleep Quality values:", df["Sleep_Quality"].unique())
print("Work Location values:", df["Work_Location"].unique())
print("\nStress values:", df["Stress_Level"].unique())
print("Productivity values:", df["Productivity_Change"].unique())
# CLEAN TEXT DATA

# Convert all relevant columns to string and clean
df["Work_Location"] = df["Work_Location"].astype(str).str.strip().str.lower()
df["Sleep_Quality"] = df["Sleep_Quality"].astype(str).str.strip().str.lower()
df["Stress_Level"] = df["Stress_Level"].astype(str).str.strip().str.lower()
df["Productivity_Change"] = df["Productivity_Change"].astype(str).str.strip().str.lower()

df["Work_Location"] = df["Work_Location"].map({
    "remote": 0,
    "hybrid": 1,
    "onsite": 2
})

df["Sleep_Quality"] = df["Sleep_Quality"].map({
    "poor": 0,
    "average": 1,
    "good": 2,
   
})

df["Stress_Level"] = df["Stress_Level"].map({
    "low": 0,
    "medium": 1,
    "high": 2
})

df["Productivity_Change"] = df["Productivity_Change"].map({
    "decrease": 0,
    "decreased": 0,
    "increase": 2,
    "increased": 2,
    "no change": 1
})


df = df.dropna(subset=[
    "Work_Location",
    "Sleep_Quality",
    "Stress_Level",
    "Productivity_Change"
])



print("\nShape AFTER dropna:", df.shape)

# features(inputs)
X = df[[
    "Work_Location",
    "Hours_Worked_Per_Week",
    "Number_of_Virtual_Meetings",
    "Work_Life_Balance_Rating",
    "Social_Isolation_Rating",
    "Sleep_Quality"
]]

# targets
y_stress = df["Stress_Level"]
y_productivity = df["Productivity_Change"]

# split
X_train, X_test, y_train_s, y_test_s, y_train_p, y_test_p = train_test_split(
    X, y_stress, y_productivity, test_size=0.2, random_state=42
)

# train the model for stress_Level & prodctivity 
model_stress=RandomForestClassifier(  n_estimators=200,
    max_depth=10,
    random_state=42
)
model_stress.fit(X_train,y_train_s)

model_productivity=RandomForestClassifier(  n_estimators=200,
    max_depth=10,
    random_state=42
)
model_productivity.fit(X_train,y_train_p)

# save the models
joblib.dump(model_stress, "stress_model.pkl")
joblib.dump(model_productivity, "productivity_model.pkl")

# evaluate the models
y_pred_s=model_stress.predict(X_test)
y_pred_p=model_productivity.predict(X_test)

# print accuracy
print("Stress Model Accuracy:", accuracy_score(y_test_s, y_pred_s))
print("productivity model accuracy:",accuracy_score(y_test_p,y_pred_p))