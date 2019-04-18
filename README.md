
# Ridge and Lasso Regression - Lab

## Introduction

In this lab, you'll practice your knowledge on Ridge and Lasso regression!

## Objectives

You will be able to:

- Use Lasso and ridge regression in Python
- Compare Lasso and Ridge with standard regression

## Housing Prices Data

Let's look at yet another house pricing data set.


```python
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

df = pd.read_csv('Housing_Prices/train.csv')
```

Look at df.info


```python
df.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 81 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    MSZoning         1460 non-null object
    LotFrontage      1201 non-null float64
    LotArea          1460 non-null int64
    Street           1460 non-null object
    Alley            91 non-null object
    LotShape         1460 non-null object
    LandContour      1460 non-null object
    Utilities        1460 non-null object
    LotConfig        1460 non-null object
    LandSlope        1460 non-null object
    Neighborhood     1460 non-null object
    Condition1       1460 non-null object
    Condition2       1460 non-null object
    BldgType         1460 non-null object
    HouseStyle       1460 non-null object
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    RoofStyle        1460 non-null object
    RoofMatl         1460 non-null object
    Exterior1st      1460 non-null object
    Exterior2nd      1460 non-null object
    MasVnrType       1452 non-null object
    MasVnrArea       1452 non-null float64
    ExterQual        1460 non-null object
    ExterCond        1460 non-null object
    Foundation       1460 non-null object
    BsmtQual         1423 non-null object
    BsmtCond         1423 non-null object
    BsmtExposure     1422 non-null object
    BsmtFinType1     1423 non-null object
    BsmtFinSF1       1460 non-null int64
    BsmtFinType2     1422 non-null object
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    Heating          1460 non-null object
    HeatingQC        1460 non-null object
    CentralAir       1460 non-null object
    Electrical       1459 non-null object
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    KitchenQual      1460 non-null object
    TotRmsAbvGrd     1460 non-null int64
    Functional       1460 non-null object
    Fireplaces       1460 non-null int64
    FireplaceQu      770 non-null object
    GarageType       1379 non-null object
    GarageYrBlt      1379 non-null float64
    GarageFinish     1379 non-null object
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    GarageQual       1379 non-null object
    GarageCond       1379 non-null object
    PavedDrive       1460 non-null object
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    PoolQC           7 non-null object
    Fence            281 non-null object
    MiscFeature      54 non-null object
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    SaleType         1460 non-null object
    SaleCondition    1460 non-null object
    SalePrice        1460 non-null int64
    dtypes: float64(3), int64(35), object(43)
    memory usage: 924.0+ KB


First, make a selection of the data by removing some of the data with `dtype = object`, this way our first model only contains **continuous features**

Make sure to remove the SalesPrice column from the predictors (which you store in `X`), then replace missing inputs by the median per feature.

Store the target in `y`.


```python
# Load necessary packages
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# remove "object"-type features and SalesPrice from `X`
features = [col for col in df.columns if df[col].dtype in [np.float64, np.int64] and col!='SalePrice']
X = df[features]

# Impute null values
for col in X:
    med = X[col].median()
    X[col].fillna(value = med, inplace = True)

# Create y
y = df.SalePrice
```

Look at the information of `X` again


```python
 X.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 1460 entries, 0 to 1459
    Data columns (total 37 columns):
    Id               1460 non-null int64
    MSSubClass       1460 non-null int64
    LotFrontage      1460 non-null float64
    LotArea          1460 non-null int64
    OverallQual      1460 non-null int64
    OverallCond      1460 non-null int64
    YearBuilt        1460 non-null int64
    YearRemodAdd     1460 non-null int64
    MasVnrArea       1460 non-null float64
    BsmtFinSF1       1460 non-null int64
    BsmtFinSF2       1460 non-null int64
    BsmtUnfSF        1460 non-null int64
    TotalBsmtSF      1460 non-null int64
    1stFlrSF         1460 non-null int64
    2ndFlrSF         1460 non-null int64
    LowQualFinSF     1460 non-null int64
    GrLivArea        1460 non-null int64
    BsmtFullBath     1460 non-null int64
    BsmtHalfBath     1460 non-null int64
    FullBath         1460 non-null int64
    HalfBath         1460 non-null int64
    BedroomAbvGr     1460 non-null int64
    KitchenAbvGr     1460 non-null int64
    TotRmsAbvGrd     1460 non-null int64
    Fireplaces       1460 non-null int64
    GarageYrBlt      1460 non-null float64
    GarageCars       1460 non-null int64
    GarageArea       1460 non-null int64
    WoodDeckSF       1460 non-null int64
    OpenPorchSF      1460 non-null int64
    EnclosedPorch    1460 non-null int64
    3SsnPorch        1460 non-null int64
    ScreenPorch      1460 non-null int64
    PoolArea         1460 non-null int64
    MiscVal          1460 non-null int64
    MoSold           1460 non-null int64
    YrSold           1460 non-null int64
    dtypes: float64(3), int64(34)
    memory usage: 422.1 KB


## Let's use this data to perform a first naive linear regression model

Compute the R squared and the MSE for both train and test set.


```python
from sklearn.metrics import mean_squared_error, mean_squared_log_error

# Split in train and test
X_train, X_test, y_train, y_test = train_test_split(X,y)

# Fit the model and print R2 and MSE for train and test
linreg = LinearRegression()
linreg.fit(X_train, y_train)

print('Training r^2:', linreg.score(X_train, y_train))
print('Testing r^2:', linreg.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, linreg.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, linreg.predict(X_test)))
```

    Training r^2: 0.804114659599992
    Testing r^2: 0.8220293726060767
    Training MSE: 1187209332.7110069
    Testing MSE: 1250121037.6196253


## Normalize your data

We haven't normalized our data, let's create a new model that uses `preprocessing.scale` to scale our predictors!


```python
from sklearn import preprocessing

# scale the data and perform train test split
X_scaled = preprocessing.scale(X)

X_train, X_test, y_train, y_test = train_test_split(X_scaled,y)
```

Perform the same linear regression on this data and print out R-squared and MSE.


```python
linreg_norm = LinearRegression()
linreg_norm.fit(X_train, y_train)
print('Training r^2:', linreg_norm.score(X_train, y_train))
print('Testing r^2:', linreg_norm.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, linreg_norm.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, linreg_norm.predict(X_test)))
```

    Training r^2: 0.8074980907068123
    Testing r^2: 0.8058185208261739
    Training MSE: 1150487580.7440414
    Testing MSE: 1416852598.1566381


## Include dummy variables

Your model hasn't included dummy variables so far: let's use the "object" variables again and create dummies


```python
# Create X_cat which contains only the categorical variables
features_cat = [col for col in df.columns if df[col].dtype in [np.object]]
X_cat = df[features_cat]

np.shape(X_cat)
```




    (1460, 43)




```python
# Make dummies
X_cat = pd.get_dummies(X_cat)
np.shape(X_cat)
```




    (1460, 252)



Merge `x_cat` together with our scaled `X` so you have one big predictor dataframe.


```python
X_all = pd.concat([pd.DataFrame(X_scaled), X_cat], axis = 1)
```

Perform the same linear regression on this data and print out R-squared and MSE.


```python
X_train, X_test, y_train, y_test = train_test_split(X_all, y)
linreg_all = LinearRegression()
linreg_all.fit(X_train, y_train)
print('Training r^2:', linreg_all.score(X_train, y_train))
print('Testing r^2:', linreg_all.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, linreg_all.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, linreg_all.predict(X_test)))
```

    Training r^2: 0.944755253171197
    Testing r^2: -4.052906884197962e+19
    Training MSE: 307167504.32054794
    Testing MSE: 3.405455332774927e+29


Notice the severe overfitting above; our training R squared is quite high, but the testing R squared is negative! Our predictions are far off. Similarly, the scale of the Testing MSE is orders of magnitude higher than that of the training.

## Perform Ridge and Lasso regression

Use all the data (normalized features and dummy categorical variables) and perform Lasso and Ridge regression for both! Each time, look at R-squared and MSE.

## Lasso

With default parameter (alpha = 1)


```python
from sklearn.linear_model import Lasso, Ridge

lasso = Lasso() 
lasso.fit(X_train, y_train)
print('Training r^2:', lasso.score(X_train, y_train))
print('Testing r^2:', lasso.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test)))
```

    Training r^2: 0.9451071623243285
    Testing r^2: 0.8178180225894447
    Training MSE: 305210846.6739859
    Testing MSE: 1530784210.5309837


With a higher regularization parameter (alpha = 10)


```python
from sklearn.linear_model import Lasso, Ridge

lasso = Lasso(alpha=10) #Lasso is also known as the L1 norm.
lasso.fit(X_train, y_train)
print('Training r^2:', lasso.score(X_train, y_train))
print('Testing r^2:', lasso.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, lasso.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, lasso.predict(X_test)))
```

    Training r^2: 0.9438669691644985
    Testing r^2: 0.8296396546959763
    Training MSE: 312106471.31972647
    Testing MSE: 1431452937.3249426


## Ridge

With default parameter (alpha = 1)


```python
from sklearn.linear_model import Lasso, Ridge

ridge = Ridge() #Lasso is also known as the L1 norm.
ridge.fit(X_train, y_train)
print('Training r^2:', ridge.score(X_train, y_train))
print('Testing r^2:', ridge.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, ridge.predict(X_test)))
```

    Training r^2: 0.9307909182663064
    Testing r^2: 0.8360653433796003
    Training MSE: 384810902.98655874
    Testing MSE: 1377461083.0345898


With default parameter (alpha = 10)


```python
from sklearn.linear_model import Lasso, Ridge

ridge = Ridge(alpha = 10) #Lasso is also known as the L1 norm.
ridge.fit(X_train, y_train)
print('Training r^2:', ridge.score(X_train, y_train))
print('Testing r^2:', ridge.score(X_test, y_test))
print('Training MSE:', mean_squared_error(y_train, ridge.predict(X_train)))
print('Testing MSE:', mean_squared_error(y_test, ridge.predict(X_test)))
```

    Training r^2: 0.9103367925088878
    Testing r^2: 0.835576679159496
    Training MSE: 498538327.2688102
    Testing MSE: 1381567084.533856


## Look at the metrics, what are your main conclusions?   

Conclusions here

## Compare number of parameter estimates that are (very close to) 0 for Ridge and Lasso


```python
print(sum(abs(ridge.coef_) < 10**(-10)))
```

    11



```python
print(sum(abs(lasso.coef_) < 10**(-10)))
```

    86


Compare with the total length of the parameter space and draw conclusions!

Lasso was very effective to essentially perform variable selection and remove about 25% of the variables from your model!


```python
len(lasso.coef_)
```




    289




```python
sum(abs(lasso.coef_) < 10**(-10))/289
```




    0.2975778546712803



## Summary

Great! You now know how to perform Lasso and Ridge regression.
