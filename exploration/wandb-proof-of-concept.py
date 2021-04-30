#%%
import numpy as np
import seaborn as sns
import pandas as pd
import wandb
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split

#%%
data = sns.load_dataset('penguins')
data = data.dropna()

X = data.select_dtypes(include='number')
feature_names = X.columns.to_numpy()
X = X.values
y, labels = pd.factorize(data['species'].values)
feature_names, labels, X.shape, y.shape

#%%
test_size = 0.25
random_state = 34

#%%
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=test_size, random_state=random_state)

#%%
clfs = [
    ('Logistic Regression', LogisticRegression(random_state=random_state)),
    ('Decision Tree', DecisionTreeClassifier(random_state=random_state)),
    ('Neural Network', MLPClassifier(random_state=random_state))
]

#%%
for name, clf in clfs:
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    y_probas = clf.predict_proba(X_test)

    wandb.init(project='visualize-penguin-logistic',
                config=clf.get_params(),
                name=name)
    wandb.sklearn.plot_classifier(clf, X_train, X_test, y_train, y_test,
        y_pred, y_probas, labels,
        model_name=type(clf).__name__,
        feature_names=list(feature_names))
    wandb.finish()
