import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import VotingClassifier, RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report, confusion_matrix



"""# Loading dataset"""

df = pd.read_csv("social_network_ads_dataset.csv")

print(df.head())

print("Dataset shape:", df.shape)

print(df.isnull().sum())

# Step 1: remove duplicate rows.
df = df.drop_duplicates().reset_index(drop=True)

# Step 2: drop the identifier column.
df = df.drop(columns=['User ID'])

# Step 3: normalize categorical text values.
df['Gender'] = df['Gender'].astype(str).str.strip().str.title()

print(df.describe())



# Step 4: build preprocessing pipelines for numeric and categorical features.

numeric_features = ['Age', 'EstimatedSalary']
categorical_features = ['Gender']

numeric_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

categorical_transformer = Pipeline([
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('encoder', OneHotEncoder(drop='first', handle_unknown='ignore'))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, numeric_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)



# Step 5: create the training pipeline and train the model.

X = df[['Gender', 'Age', 'EstimatedSalary']]
y = df['Purchased']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, random_state=42, stratify=y
)

log_reg = LogisticRegression(random_state=42, max_iter=500)
rf = RandomForestClassifier(random_state=42)
gb = GradientBoostingClassifier(random_state=42)

voting_clf = VotingClassifier(
    estimators=[('lr', log_reg), ('rf', rf), ('gb', gb)],
    voting='soft'
)

pipe = Pipeline([
    ('preprocessor', preprocessor),
    ('model', voting_clf)
])

pipe.fit(X_train, y_train)




"""# Cross-validation"""

cv_scores = cross_val_score(pipe, X, y, cv=10, scoring='accuracy')
print("CV Scores:", cv_scores)
print("Average Accuracy:", cv_scores.mean())
print("Std Deviation:", cv_scores.std())


"""# Primary Model Selection

Chosen Model: Voting Ensemble
1.Combines Logistic Regression, Random Forest, and Gradient Boosting.
2.Achieved highest accuracy (0.92) and balanced precision/recall in earlier tests.



#  Hyperparameter Tuning"""
param_grid = {
    'model__lr__C': [0.1, 1, 10],
    'model__lr__solver': ['liblinear', 'lbfgs'],
    'model__rf__n_estimators': [100, 200],
    'model__rf__max_depth': [None, 5, 10],
    'model__gb__n_estimators': [100, 200],
    'model__gb__learning_rate': [0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(pipe, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train, y_train)

print("Best Parameters:", grid_search.best_params_)
print("Best CV Accuracy:", grid_search.best_score_)



"""Best Model Selection"""

best_model = grid_search.best_estimator_



"""# Model Performance Evaluation"""

y_pred = best_model.predict(X_test)

print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))



"""# Saving Model"""

import pickle
filename = "bestmodel.pkl"

with open( filename, "wb" ) as file:
  pickle.dump( best_model, file )

with open( "bestmodel.pkl", "rb" ) as file:
  bm_loaded_model = pickle.load(file)

bm_loaded_model.predict(X_test)

print("Successfully loaded and used the model")