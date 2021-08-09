## Data Precessing

Once we know our data briefly, we can preprocess our data, which may fast our training, improve model performance and so onAssuming that our data is stored in **Pandas Dataframe format**, and the following code is based on it.



### Transform Continuous Value into Discrete Value

#### .cut()
&emsp;
We can use the ``pd.cut()`` function to create a new category, 'income_cat', which is dividedan 'median_income' category into 5 groups (labeled from 1 to 5): 
category 1 ranges from 0 to 1.5 (i.e., less than $15,000), 
category 2 from 1.5 to 3, and so on


```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
                               
housing["income_cat"].value_counts()
```


    3    7236
    2    6581
    4    3639
    5    2362
    1     822
    Name: income_cat, dtype: int64


```python
housing["income_cat"].hist()
```

![image](https://github.com/dynotw/Basic-Machine-Learning-Code/blob/main/Machine%20Learning/Data/Preview/02_end_to_end_machine_learning_project_31_1.png)




### Data Cleaning

#### Missing value
Most Machine Learning algorithms cannot work with missing features, so let’s create a few functions to take care of them. Assume that total_bedrooms attribute has some missing values, so let’s fix this. You have three options: 

* Get rid of the corresponding districts.
* Get rid of the whole attribute.
* Set the values to some value (zero, the mean, the median, etc.).


You can accomplish these easily using DataFrame’s ``.dropna()``, ``.drop()``, and ``.fillna()`` methods:


Have a quick view, which samples, which have missing informations, look like

```python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>NaN</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>NaN</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>NaN</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>NaN</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>NaN</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



&nbsp;
* Method 1, ``.dropna()``

``.dropna()`` is to drop all rows, which have missing information.

```python
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>


&nbsp;
* Method 2, ``.drop()``

``.drop()`` is to drop the column, according to the column names.

```python
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>


&nbsp;
* Method 3, ``.fillna()``

``.fillna()`` is totally different from the above two methods. ``.fillna()`` is to fill the missing information, rather than drop them. **sklearn provides a more powerful class ``SimpleImputer`` to deal with missing values

```python
median = housing["total_bedrooms"].median() # Use median value to fill those missing data
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
sample_incomplete_rows
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>433.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>433.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>433.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>433.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>433.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>


&nbsp;
* Method 4, ``sklearn.impute.SimpleImputer``

```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median") # Choose median to fill missing value
```

Remove the text attribute because median can only be calculated on numerical attributes:


```python
housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
```


Now you can fit the imputer instance to the training data using the fit() method:

```python
imputer.fit(housing_num) 
# On the training set, we usually use ``.fit_transform()``, which is equivalent to calling fit() and then transform() (but sometimes fit_transform() is optimized and runs much faster). 
# On valdation & test set, we only use ``.transform()``, because we want transform on validation & test set is exactly same as training set.

imputer.statistics_
```

    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])

Transform the training set:


```python
# use this “trained” imputer to transform the training set by replacing missing values by the learned medians:
X = imputer.transform(housing_num)

# The X is a plain NumPy array, so we may need to put it back into a Pandas Dataframe
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
                          
# Let's check whether missing value has been filled
housing_tr.loc[sample_incomplete_rows.index.values]
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>4629</th>
      <td>-118.30</td>
      <td>34.07</td>
      <td>18.0</td>
      <td>3759.0</td>
      <td>433.0</td>
      <td>3296.0</td>
      <td>1462.0</td>
      <td>2.2708</td>
    </tr>
    <tr>
      <th>6068</th>
      <td>-117.86</td>
      <td>34.01</td>
      <td>16.0</td>
      <td>4632.0</td>
      <td>433.0</td>
      <td>3038.0</td>
      <td>727.0</td>
      <td>5.1762</td>
    </tr>
    <tr>
      <th>17923</th>
      <td>-121.97</td>
      <td>37.35</td>
      <td>30.0</td>
      <td>1955.0</td>
      <td>433.0</td>
      <td>999.0</td>
      <td>386.0</td>
      <td>4.6328</td>
    </tr>
    <tr>
      <th>13656</th>
      <td>-117.30</td>
      <td>34.05</td>
      <td>6.0</td>
      <td>2155.0</td>
      <td>433.0</td>
      <td>1039.0</td>
      <td>391.0</td>
      <td>1.6675</td>
    </tr>
    <tr>
      <th>19252</th>
      <td>-122.79</td>
      <td>38.48</td>
      <td>7.0</td>
      <td>6837.0</td>
      <td>433.0</td>
      <td>3468.0</td>
      <td>1405.0</td>
      <td>3.1662</td>
    </tr>
  </tbody>
</table>
</div>


&emsp;
### Deal with Text & Categorical Atributes (Discrete / Sparse)

In our dataset, there is a feature, named `ocean_proximity`, which is a text feature. We know machine learning model can't learn text information, so we need to transfrom them into numerical information. In `` scikit-learn``, we have two classes, ``sklearn.preprocessing.OrdinalEncoder`` & ``sklearn.preprocessing.OneHotEncoder``

&nbsp;
* Method 1, ``sklearn.preprocessing.OrdinalEncoder``

This method is useful in features, whose number of unique value is small. 他的本质就是用数字去表示每一种text value，即将text value 转变成 numerical value。

```python
# Pick up the columns, which is text information
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>NEAR OCEAN</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>19480</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>8879</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>13685</th>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>4937</th>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>4861</th>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



```python
from sklearn.preprocessing import OrdinalEncoder

ordinal_encoder = OrdinalEncoder()
housing_cat_encoded = ordinal_encoder.fit_transform(housing_cat)
housing_cat_encoded[:10]
```

    array([[0.],
           [0.],
           [4.],
           [1.],
           [0.],
           [1.],
           [0.],
           [1.],
           [0.],
           [0.]])


You can get the list of categories using the ``.categories_`` instance variable. It is a list containing a 1D array of categories for each categorical attribute (in this case, a list containing a single array since there is just one categorical attribute). 这个categories是按照顺序的，0 -> '<1H OCEAN'; 1 -> 'INLAND'; 2 -> 'ISLAND' and so on.

```python
ordinal_encoder.categories_
```

    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]


``OrdinalEncoder`` has one problem that ML algorithms will assume that two nearby values are more similar than two distant values. (0 is similar to 1, not 2) This may be fine in some cases (e.g., for ordered categories such as “bad”, “average”, “good”, “excellent”), but it is obviously not the case for the ocean_proximity column. For example, categories 0, (<1H OCEAN) and 4, (NEAR OCEAN) are clearly more similar than categories 0, (<1H OCEAN) and 1, (INLAND).
** To solve this problem, we can use another scikit-learn class, named OneHotEncoder**


&nbsp;
* Method 2, ``sklearn.preprocessing.OneHotEncoder``

```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
```

By default, the `OneHotEncoder` class returns a SciPy sparse matrix, but we can convert it to a Numpy array if needed by calling the `toarray()` method:


```python
housing_cat_1hot.toarray()
```


    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])



Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`: ``cat_encoder = OneHotEncoder(sparse=False)``



Once again, you can get the list of categories using the encoder’s ``categories_`` instance variable:

```python
cat_encoder.categories_
```


    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



&emsp;
### Custom Transform Method

#### 1. Custom own Transformer
Although Scikit-Learn provides many useful transformers, we also can write our own custom transformers for cleanup operations or combining specific attributes. All you need is to create a class and implement three methods: ``.fit()`` (returning self), ``.transform()``, and ``.fit_transform()``. You can get the last one for free by simply adding ``TransformerMixin`` as a base class. Also, if you add ``BaseEstimator`` as a base class (and avoid ``*args`` and ``**kargs`` in your constructor) you will get two extra methods (``get_params()`` and ``set_params()``) that will be useful for automatic hyperparameter tuning.


Let's create a custom transformer to create extra attributes:


```python
from sklearn.base import BaseEstimator, TransformerMixin

# column index, these are global variables. Now just create these global variable with random values, and we will assign actual values to these variables later 
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6


# This custom Transformer is to add two features into our data
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X): # X is a 2D numpy array, not pandas.Dataframe
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            
            # np.c_[] is to merge arrays by rows, which requires that all arrays have same number of rows.
            # By contrast, is np.r_[] , which merge arrays by columns
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)


col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [housing.columns.get_loc(c) for c in col_names] # get the column indices
```

Also, `housing_extra_attribs` is a NumPy array, we've lost the column names (unfortunately, that's a problem with Scikit-Learn). To recover a `DataFrame`, you could run this:


```python
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()
```


<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>longitude</th>
      <th>latitude</th>
      <th>housing_median_age</th>
      <th>total_rooms</th>
      <th>total_bedrooms</th>
      <th>population</th>
      <th>households</th>
      <th>median_income</th>
      <th>ocean_proximity</th>
      <th>rooms_per_household</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>17606</th>
      <td>-121.89</td>
      <td>37.29</td>
      <td>38.0</td>
      <td>1568.0</td>
      <td>351.0</td>
      <td>710.0</td>
      <td>339.0</td>
      <td>2.7042</td>
      <td>&lt;1H OCEAN</td>
      <td>4.625369</td>
      <td>2.094395</td>
    </tr>
    <tr>
      <th>18632</th>
      <td>-121.93</td>
      <td>37.05</td>
      <td>14.0</td>
      <td>679.0</td>
      <td>108.0</td>
      <td>306.0</td>
      <td>113.0</td>
      <td>6.4214</td>
      <td>&lt;1H OCEAN</td>
      <td>6.00885</td>
      <td>2.707965</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.2</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>NEAR OCEAN</td>
      <td>4.225108</td>
      <td>2.025974</td>
    </tr>
    <tr>
      <th>3230</th>
      <td>-119.61</td>
      <td>36.31</td>
      <td>25.0</td>
      <td>1847.0</td>
      <td>371.0</td>
      <td>1460.0</td>
      <td>353.0</td>
      <td>1.8839</td>
      <td>INLAND</td>
      <td>5.232295</td>
      <td>4.135977</td>
    </tr>
    <tr>
      <th>3555</th>
      <td>-118.59</td>
      <td>34.23</td>
      <td>17.0</td>
      <td>6592.0</td>
      <td>1525.0</td>
      <td>4459.0</td>
      <td>1463.0</td>
      <td>3.0347</td>
      <td>&lt;1H OCEAN</td>
      <td>4.50581</td>
      <td>3.047847</td>
    </tr>
  </tbody>
</table>
</div>


&emsp;
#### 2. Custom Transformation Pipeline

Sometimes we want to do a series of different transformations on our data orderly, it's very tedious, if we implement these transform one by one. Therefore, we create a Pipeline class instance. **补充：everything is object, even class itself. 类可以是对象，类实例化后的出来的是此类的实例对象。** Then this instance will do these transformation orderly and automatically.


Now let's build a pipeline for preprocessing (transforming) on the numerical attributes:

* 1. ``SimpleImputer``, by median
* 2. ``CombinedAttributesAdder``, which is our custom transformer defined early.
* 3. ``StanardSclaer``, 注意区分不同的Scaling 技术 like ``MinMaxScaler``, ``Normalizer`` and so on


**需要注意Pipeline是对所有feature同时进行处理，如果不同features（columns）需要进行不同的transform，应该选用下面的``ColumnTransformer``**
因为此Pipeline中的transformation都是针对numerical value，所以num_pipeline只能用于housing_num, which removes text information, instead of housing, which includes text feature, ['ocean_proximity'] 。 （否则就报错了）

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# 'imputer', 'attribs_adder' & 'std_scaler' just names, we can name them whatever we like
num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

# Same as above, train set: .fit_transform();    validation & test set: .transform() 
housing_num_tr = num_pipeline.fit_transform(housing_num)
```


**Notice** We alsoc can put predictor into Pipeline, not only transformer. For example:

```python
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)
```


&emsp;
#### 3. Custom Column Transformer

Sometimes we need to do different transformation on different features(columns), this time we can use ``ColumnTransformer`` to deal with numerical and text information respectively.



```python
from sklearn.compose import ColumnTransformer

# Get the index or column names of features (columns)
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        # We divide into two parts: numerical information and text information
        ("num", num_pipeline, num_attribs), # Each tuple has three elements: name, transformer, columns index or name
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)

housing_prepared.shape
```


    (16512, 16)

