from preprocess import prepare_data

X_train, X_test, y_train, y_test = prepare_data()

print(X_train.shape)
print(X_test.shape)