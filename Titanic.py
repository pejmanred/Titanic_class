import pandas as pd
trainD=pd.read_csv('train.csv')  #Read the train data
testD=pd.read_csv('test.csv')    #read the test data

trainD.info()
trainD.describe()   


from sklearn.base import BaseEstimator, TransformerMixin

# A class to select numerical or categorical columns 
# since Scikit-Learn doesn't handle DataFrames yet
class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names]
    
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer # Scikit-Learn 0.20+


num_pipeline = Pipeline([
        ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
        ("imputer", SimpleImputer(strategy="median")),
    ])

num_pipeline.fit_transform(trainD)

# Inspired from stackoverflow.com/questions/25239958
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X],
                                        index=X.columns)
        return self
    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)
    
from sklearn.preprocessing import OneHotEncoder

cat_pipeline = Pipeline([
        ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
        ("imputer", MostFrequentImputer()),
        ("cat_encoder", OneHotEncoder(sparse=False)),
    ])

cat_pipeline.fit_transform(trainD)

from sklearn.pipeline import FeatureUnion
preprocess_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", num_pipeline),
        ("cat_pipeline", cat_pipeline),
    ])

x_train = preprocess_pipeline.fit_transform(trainD)
y_train = trainD["Survived"]
x_test = preprocess_pipeline.transform(testD)

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV 

forest_clf = RandomForestClassifier(n_estimators=100, random_state=42)
forest_scores = cross_val_score(forest_clf, x_train, y_train, cv=10)
forest_scores.mean()

from sklearn.metrics import make_scorer
from sklearn.metrics import accuracy_score

scoring = {'AUC': 'roc_auc', 'Accuracy': make_scorer(accuracy_score)}
parameters = {'n_estimators':range(10,100,120), 'criterion':('gini', 'entropy'),'max_features':('auto', 'sqrt', 'log2')}
gs=GridSearchCV(forest_clf,parameters,scoring=scoring,refit='AUC', return_train_score=True,cv=3)
gs.fit(x_train,y_train)
print(gs.best_score_)

rf_predic=gs.best_estimator_.predict(x_test)



