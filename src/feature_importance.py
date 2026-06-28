from pandas import DataFrame

def show_feature_importance(model, X_train):

    importance = model.feature_importances_

    df = DataFrame({
        "Feature": X_train.columns,
        "Importance": importance
    })

    df = df.sort_values(
        by="Importance",
        ascending=False
    )

    print(df.head(20))

    df.to_csv(
        "results/feature_importance.csv",
        index=False
    )