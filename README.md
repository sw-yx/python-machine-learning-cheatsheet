# python-machine-learning-cheatsheet

cheatsheet

## Useful Top level settings

```python
import warnings
warnings.simplefilter("ignore")
import matplotlib.pyplot as plt
plt.style.use('ggplot')
# %matplotlib inline
np.random.seed(55)
```

## EDA

```py
## peek at the top
dataset.head()
## checking for nulls
dataset.isnull().sum()
## describe count/unique/top/freq
dataset.describe()
```

## Splitting Data

```py
X=dataset.drop('class',axis=1) #Predictors
y=dataset['class'] #Response

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

## Encoding Categorical data

This does ordinal encoding - strings to numbers. not great when the number order doesnt mean anything but can be helpful.

```py
from sklearn.preprocessing import LabelEncoder
Encoder_X = LabelEncoder() 
for col in X.columns:
    X[col] = Encoder_X.fit_transform(X[col])
Encoder_y=LabelEncoder()
y = Encoder_y.fit_transform(y)
```


`np.where` custom encoding

```py
y_test = np.where(mushrooms_tstY == "p", 1, 0)
```


One-hot encoding

```py
X=pd.get_dummies(X,columns=X.columns,drop_first=True)
```

## Prepping data

Feature Scaling

```py
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()

X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)
```

PCA

```py
from sklearn.decomposition import PCA
pca = PCA(n_components=2)

X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
```


## Visualization

```py
def visualization_train(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_train, y_train
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                     np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
             alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Training Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
def visualization_test(model):
    sns.set_context(context='notebook',font_scale=2)
    plt.figure(figsize=(16,9))
    from matplotlib.colors import ListedColormap
    X_set, y_set = X_test, y_test
    X1, X2 = np.meshgrid(np.arange(start = X_set[:, 0].min() - 1, stop = X_set[:, 0].max() + 1, step = 0.01),
                         np.arange(start = X_set[:, 1].min() - 1, stop = X_set[:, 1].max() + 1, step = 0.01))
    plt.contourf(X1, X2, classifier.predict(np.array([X1.ravel(), X2.ravel()]).T).reshape(X1.shape),
                 alpha = 0.6, cmap = ListedColormap(('red', 'green')))
    plt.xlim(X1.min(), X1.max())
    plt.ylim(X2.min(), X2.max())
    for i, j in enumerate(np.unique(y_set)):
        plt.scatter(X_set[y_set == j, 0], X_set[y_set == j, 1],
                    c = ListedColormap(('red', 'green'))(i), label = j)
    plt.title("%s Test Set" %(model))
    plt.xlabel('PC 1')
    plt.ylabel('PC 2')
    plt.legend()
```

## ANN

Training

```py
import keras
from keras.models import Sequential
from keras.layers import Dense
classifier = Sequential()
classifier.add(Dense(8, kernel_initializer='uniform', activation= 'relu', input_dim = 2))
classifier.add(Dense(6, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(5, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(4, kernel_initializer='uniform', activation= 'relu'))
classifier.add(Dense(1, kernel_initializer= 'uniform', activation= 'sigmoid'))
classifier.compile(optimizer= 'adam',loss='binary_crossentropy', metrics=['accuracy'])
classifier.fit(X_train,y_train,batch_size=10,epochs=100)
```

Predicting

```py
y_pred=classifier.predict(X_test)
y_pred=(y_pred>0.5)
```

Confusion & Classification Report

```py
from sklearn.metrics import confusion_matrix,classification_report
print(confusion_matrix(y_test, y_pred))
print(classification_report(y_test, y_pred))
```

Visualization

```py
visualization_train(model='ANN')
visualization_test(model='ANN')
```

Model Evaluation

```py
from sklearn.model_selection import cross_val_predict, cross_val_score
from sklearn.metrics import confusion_matrix,classification_report,accuracy_score
def print_score(classifier,X_train,y_train,X_test,y_test,train=True):
    if train == True:
        print("Training results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_train,classifier.predict(X_train))))
        print('Classification Report:\n{}\n'.format(classification_report(y_train,classifier.predict(X_train))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_train,classifier.predict(X_train))))
        res = cross_val_score(classifier, X_train, y_train, cv=10, n_jobs=-1, scoring='accuracy')
        print('Average Accuracy:\t{0:.4f}\n'.format(res.mean()))
        print('Standard Deviation:\t{0:.4f}'.format(res.std()))
    elif train == False:
        print("Test results:\n")
        print('Accuracy Score: {0:.4f}\n'.format(accuracy_score(y_test,classifier.predict(X_test))))
        print('Classification Report:\n{}\n'.format(classification_report(y_test,classifier.predict(X_test))))
        print('Confusion Matrix:\n{}\n'.format(confusion_matrix(y_test,classifier.predict(X_test))))
```

## Support Vector Classification

```py
from sklearn.svm import SVC
classifier = SVC(kernel='rbf',random_state=42)

classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
visualization_train('SVC')
visualization_test('SVC')
```

## KNN

```py
from sklearn.neighbors import KNeighborsClassifier as KNN

classifier = KNN()
classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
```

## Decision Tree

```py
from sklearn.tree import DecisionTreeClassifier as DT

classifier = DT(criterion='entropy',random_state=42)
classifier.fit(X_train,y_train)
print_score(classifier,X_train,y_train,X_test,y_test,train=True)
print_score(classifier,X_train,y_train,X_test,y_test,train=False)
```

## Nested Cross Eval

```py
param_grid = [{'C': np.logspace(-3, 3, 10)}]

grid_search = GridSearchCV(
    estimator=LogisticRegression(),
    param_grid=param_grid,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=42),
    scoring='f1',
    n_jobs=-1
)

scores = cross_val_score(
    estimator=grid_search,
    X=X_std,
    y=y.enc,
    cv=StratifiedShuffleSplit(n_splits=10, test_size=0.2, random_state=0),
    scoring='f1',
    n_jobs=-1
)
```

## Pipelines

- https://www.kaggle.com/baghern/a-deep-dive-into-sklearn-pipelines
- https://medium.com/bigdatarepublic/integrating-pandas-and-scikit-learn-with-pipelines-f70eb6183696
- http://michelleful.github.io/code-blog/2015/06/20/pipelines/ (approachable intro) related to https://www.youtube.com/watch?v=0UWXCAYn8rk (simple tutorial)
- http://zacstewart.com/2014/08/05/pipelines-of-featureunions-of-pipelines.html making custom transformers
- https://iaml.it/blog/optimizing-sklearn-pipelines what it says
- https://www.kdnuggets.com/2017/12/managing-machine-learning-workflows-scikit-learn-pipelines-part-1.html (non gridsearch comparison of pipelines)

## SKLearn Transformers

```py
from sklearn.base import TransformerMixin, BaseEstimator
class MyTransformer(TransformerMixin, BaseEstimator):
    """Recommended signature for a custom transformer.

    Inheriting from TransformerMixin gives you fit_transform

    Inheriting from BaseEstimator gives you grid-searchable params.
    """
    def __init__(self):
        """If you need to parameterize your transformer,
        set the args here.

        Inheriting from BaseEstimator introduces the constraint
        that the args all be named keyword args, no positional 
        args or **kwargs.
        """
        pass
    def fit(self, X, y):
        """Recommended signature for custom transformer's
        fit method.

        Set state here with whatever information
        is needed to transform later.

        In some cases fit may do nothing. For example transforming 
        degrees Fahrenheit to Kelvin, requires no state.

        You can use y here, but won't have access to it in transform.
        """
        #You have to return self, so we can chain!
        
        # do stuff to self.xyz or pass
        # e.g. self.mean = np.mean(X, axis=0)
        # e.g. self.std = np.std(X, axis=0)
        return self
    def transform(self, X):
        """Recommended signature for custom transformer's
        transform method.

        Transform some X data, optionally using state set in fit. This X
        may be the same X passed to fit, but it may also be new data,
        as in the case of a CV dataset. Both are treated the same.
        """
        #Do transforms.
        #T = X.copy()
        #T -= self.mean
        #T /= self.scale
        return T
```
from https://github.com/zipfian/pipelines_and_featureunions/blob/master/pipelines_and_featureunions.ipynb

`FunctionTransformer`: simple mapper transformer

```py
from sklearn.preprocessing import FunctionTransformer
from sklearn.pipeline import FeatureUnion

X = np.random.random((100,1))

fu = FeatureUnion([('sqrt',FunctionTransformer(np.sqrt)),
                  ('identity',FunctionTransformer()), ## None works!
                  ('square', FunctionTransformer(lambda x: x**2))])
print(X[:5])
print(fu.fit_transform(X)[:5])
```

- 	sklearn.preprocessing.OneHotEncoder
- 	sklearn.preprocessing.Imputer
- 	sklearn.preprocessing.CountVectorizer


## GridSearchCV

- https://gist.github.com/amberjrivera/8c5c145516f5a2e894681e16a8095b5c

## Naive Bayes

http://zacstewart.com/2014/01/10/building-a-language-identifier.html

## Usecase breakdowns

- spam http://zacstewart.com/2015/04/28/document-classification-with-scikit-learn.html



## References

- https://www.kaggle.com/raghuchaudhary/mushroom-classification/comments
- http://inmachineswetrust.com/posts/mushroom-classification/
