import joblib

# Load the model that you just saved
lr = joblib.load('svm.pkl')

# Saving the data columns from training
model_columns = list(x.columns)
joblib.dump(model_columns, 'model_columns.pkl')
print("Models columns dumped!")