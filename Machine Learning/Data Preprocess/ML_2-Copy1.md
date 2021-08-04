### Import Necessary Packages


```python
import os
import tarfile
import urllib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
```


```python
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
HOUSING_PATH = os.path.join("datasets", "housing")
HOUSING_URL = DOWNLOAD_ROOT + "datasets/housing/housing.tgz"

# Download a dataset about housing 
def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH): 
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


# Use pandas.Dataframe to load housig data
def load_housing_data(housing_path=HOUSING_PATH): 
    csv_path = os.path.join(housing_path, "housing.csv") 
    return pd.read_csv(csv_path)
```

### Set a dataset about housing dataset


```python
fetch_housing_data()
housing = load_housing_data()
housing.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>-122.23</td>
      <td>37.88</td>
      <td>41.0</td>
      <td>880.0</td>
      <td>129.0</td>
      <td>322.0</td>
      <td>126.0</td>
      <td>8.3252</td>
      <td>452600.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>1</th>
      <td>-122.22</td>
      <td>37.86</td>
      <td>21.0</td>
      <td>7099.0</td>
      <td>1106.0</td>
      <td>2401.0</td>
      <td>1138.0</td>
      <td>8.3014</td>
      <td>358500.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>2</th>
      <td>-122.24</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1467.0</td>
      <td>190.0</td>
      <td>496.0</td>
      <td>177.0</td>
      <td>7.2574</td>
      <td>352100.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>3</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1274.0</td>
      <td>235.0</td>
      <td>558.0</td>
      <td>219.0</td>
      <td>5.6431</td>
      <td>341300.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>4</th>
      <td>-122.25</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>1627.0</td>
      <td>280.0</td>
      <td>565.0</td>
      <td>259.0</td>
      <td>3.8462</td>
      <td>342200.0</td>
      <td>NEAR BAY</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.info()
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20640 entries, 0 to 20639
    Data columns (total 10 columns):
     #   Column              Non-Null Count  Dtype  
    ---  ------              --------------  -----  
     0   longitude           20640 non-null  float64
     1   latitude            20640 non-null  float64
     2   housing_median_age  20640 non-null  float64
     3   total_rooms         20640 non-null  float64
     4   total_bedrooms      20433 non-null  float64
     5   population          20640 non-null  float64
     6   households          20640 non-null  float64
     7   median_income       20640 non-null  float64
     8   median_house_value  20640 non-null  float64
     9   ocean_proximity     20640 non-null  object 
    dtypes: float64(9), object(1)
    memory usage: 1.6+ MB



```python
housing["ocean_proximity"].value_counts()
```




    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64




```python
housing.describe()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20433.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
      <td>20640.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.569704</td>
      <td>35.631861</td>
      <td>28.639486</td>
      <td>2635.763081</td>
      <td>537.870553</td>
      <td>1425.476744</td>
      <td>499.539680</td>
      <td>3.870671</td>
      <td>206855.816909</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.003532</td>
      <td>2.135952</td>
      <td>12.585558</td>
      <td>2181.615252</td>
      <td>421.385070</td>
      <td>1132.462122</td>
      <td>382.329753</td>
      <td>1.899822</td>
      <td>115395.615874</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>2.000000</td>
      <td>1.000000</td>
      <td>3.000000</td>
      <td>1.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.930000</td>
      <td>18.000000</td>
      <td>1447.750000</td>
      <td>296.000000</td>
      <td>787.000000</td>
      <td>280.000000</td>
      <td>2.563400</td>
      <td>119600.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.490000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2127.000000</td>
      <td>435.000000</td>
      <td>1166.000000</td>
      <td>409.000000</td>
      <td>3.534800</td>
      <td>179700.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.710000</td>
      <td>37.000000</td>
      <td>3148.000000</td>
      <td>647.000000</td>
      <td>1725.000000</td>
      <td>605.000000</td>
      <td>4.743250</td>
      <td>264725.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6445.000000</td>
      <td>35682.000000</td>
      <td>6082.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing.hist(bins = 100, figsize = (20,15))
plt.show()
```


    
![png](ML_2-Copy1_files/ML_2-Copy1_8_0.png)
    


### Make train and test set


```python
def split_train_test(data, test_ratio):
    # 生成打乱后的index permutaion， 用于后续生成train & test set
    shuffled_indices = np.random.permutation(len(data))
    # Transfer into integer
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    train_set = data.iloc[train_indices]
    test_set = data.iloc[test_indices]
    return train_set , test_set
```


```python
train_set , test_set = split_train_test(housing, 0.3)
print(len(train_set), len(test_set))
train_set.head()

```

    14448 6192





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>14512</th>
      <td>-117.16</td>
      <td>32.91</td>
      <td>5.0</td>
      <td>1619.0</td>
      <td>272.0</td>
      <td>1063.0</td>
      <td>296.0</td>
      <td>6.0891</td>
      <td>214600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9636</th>
      <td>-121.02</td>
      <td>36.94</td>
      <td>33.0</td>
      <td>1541.0</td>
      <td>313.0</td>
      <td>880.0</td>
      <td>272.0</td>
      <td>2.5074</td>
      <td>117700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>10530</th>
      <td>-117.70</td>
      <td>33.57</td>
      <td>9.0</td>
      <td>1204.0</td>
      <td>355.0</td>
      <td>469.0</td>
      <td>293.0</td>
      <td>3.6196</td>
      <td>119900.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>8048</th>
      <td>-118.17</td>
      <td>33.84</td>
      <td>45.0</td>
      <td>1853.0</td>
      <td>328.0</td>
      <td>945.0</td>
      <td>320.0</td>
      <td>5.0787</td>
      <td>219200.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>8508</th>
      <td>-118.31</td>
      <td>33.88</td>
      <td>32.0</td>
      <td>2421.0</td>
      <td>671.0</td>
      <td>1491.0</td>
      <td>587.0</td>
      <td>3.5644</td>
      <td>242300.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
# Above we define a train & test set generation fucntion by ourselves, actually we can use certain existed function 
# for example split_train_set() from sklearn

from sklearn.model_selection import train_test_split
train_set, test_set = train_test_split(housing, test_size = 0.3, random_state = 42)
```


```python
print(len(train_set), len(test_set))

```

    14448 6192



```python
from sklearn.model_selection import StratifiedShuffleSplit

# Split train & test set, based on median_income distribution
housing["income_cat"] = pd.cut(housing["median_income"], bins=[0,1.5,3,4.5,6,np.inf], labels = [1,2,3,4,5])

split = StratifiedShuffleSplit(n_splits = 1, test_size = 0.2, random_state =42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]

strat_train_set.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
      <th>ocean_proximity</th>
      <th>income_cat</th>
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
      <td>286600.0</td>
      <td>&lt;1H OCEAN</td>
      <td>2</td>
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
      <td>340600.0</td>
      <td>&lt;1H OCEAN</td>
      <td>5</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>196900.0</td>
      <td>NEAR OCEAN</td>
      <td>2</td>
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
      <td>46300.0</td>
      <td>INLAND</td>
      <td>2</td>
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
      <td>254500.0</td>
      <td>&lt;1H OCEAN</td>
      <td>3</td>
    </tr>
  </tbody>
</table>
</div>




```python
# drop "income_cat"
strat_train_set = strat_train_set.drop("income_cat", axis = 1)
strat_test_set = strat_test_set.drop("income_cat", axis = 1)
```


```python
#strat_train_set.info()
strat_train_set.head()
#strat_test_set.info()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
      <th>ocean_proximity</th>
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
      <td>286600.0</td>
      <td>&lt;1H OCEAN</td>
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
      <td>340600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>196900.0</td>
      <td>NEAR OCEAN</td>
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
      <td>46300.0</td>
      <td>INLAND</td>
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
      <td>254500.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
  </tbody>
</table>
</div>



### Explore Train Dataset


```python
train_set_temp = strat_train_set.copy()
```


```python
train_set_temp.corr()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>longitude</th>
      <td>1.000000</td>
      <td>-0.924478</td>
      <td>-0.105848</td>
      <td>0.048871</td>
      <td>0.076598</td>
      <td>0.108030</td>
      <td>0.063070</td>
      <td>-0.019583</td>
      <td>-0.047432</td>
    </tr>
    <tr>
      <th>latitude</th>
      <td>-0.924478</td>
      <td>1.000000</td>
      <td>0.005766</td>
      <td>-0.039184</td>
      <td>-0.072419</td>
      <td>-0.115222</td>
      <td>-0.077647</td>
      <td>-0.075205</td>
      <td>-0.142724</td>
    </tr>
    <tr>
      <th>housing_median_age</th>
      <td>-0.105848</td>
      <td>0.005766</td>
      <td>1.000000</td>
      <td>-0.364509</td>
      <td>-0.325047</td>
      <td>-0.298710</td>
      <td>-0.306428</td>
      <td>-0.111360</td>
      <td>0.114110</td>
    </tr>
    <tr>
      <th>total_rooms</th>
      <td>0.048871</td>
      <td>-0.039184</td>
      <td>-0.364509</td>
      <td>1.000000</td>
      <td>0.929379</td>
      <td>0.855109</td>
      <td>0.918392</td>
      <td>0.200087</td>
      <td>0.135097</td>
    </tr>
    <tr>
      <th>total_bedrooms</th>
      <td>0.076598</td>
      <td>-0.072419</td>
      <td>-0.325047</td>
      <td>0.929379</td>
      <td>1.000000</td>
      <td>0.876320</td>
      <td>0.980170</td>
      <td>-0.009740</td>
      <td>0.047689</td>
    </tr>
    <tr>
      <th>population</th>
      <td>0.108030</td>
      <td>-0.115222</td>
      <td>-0.298710</td>
      <td>0.855109</td>
      <td>0.876320</td>
      <td>1.000000</td>
      <td>0.904637</td>
      <td>0.002380</td>
      <td>-0.026920</td>
    </tr>
    <tr>
      <th>households</th>
      <td>0.063070</td>
      <td>-0.077647</td>
      <td>-0.306428</td>
      <td>0.918392</td>
      <td>0.980170</td>
      <td>0.904637</td>
      <td>1.000000</td>
      <td>0.010781</td>
      <td>0.064506</td>
    </tr>
    <tr>
      <th>median_income</th>
      <td>-0.019583</td>
      <td>-0.075205</td>
      <td>-0.111360</td>
      <td>0.200087</td>
      <td>-0.009740</td>
      <td>0.002380</td>
      <td>0.010781</td>
      <td>1.000000</td>
      <td>0.687160</td>
    </tr>
    <tr>
      <th>median_house_value</th>
      <td>-0.047432</td>
      <td>-0.142724</td>
      <td>0.114110</td>
      <td>0.135097</td>
      <td>0.047689</td>
      <td>-0.026920</td>
      <td>0.064506</td>
      <td>0.687160</td>
      <td>1.000000</td>
    </tr>
  </tbody>
</table>
</div>




```python
train_set_temp["room_per_household"] = train_set_temp["total_rooms"] / train_set_temp["households"]
train_set_temp["bedrooms_per_room"] = train_set_temp["total_bedrooms"]/train_set_temp["total_rooms"]
train_set_temp["population_per_household"] = train_set_temp["population"]/train_set_temp["households"]
```


```python
# Use one-hot encoder from sklearn
from sklearn.preprocessing import OneHotEncoder
one_hot_encoder = OneHotEncoder()
ocean_pro_onehot = one_hot_encoder.fit_transform(train_set_temp[["ocean_proximity"]])
ocean_pro_onehot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>




```python
print(type(train_set_temp[["ocean_proximity"]]))
print(type(train_set_temp["ocean_proximity"]))
```

    <class 'pandas.core.frame.DataFrame'>
    <class 'pandas.core.series.Series'>



```python
train_set_temp.values.shape
train_set_temp.info()

```

    <class 'pandas.core.frame.DataFrame'>
    Int64Index: 16512 entries, 17606 to 15775
    Data columns (total 13 columns):
     #   Column                    Non-Null Count  Dtype  
    ---  ------                    --------------  -----  
     0   longitude                 16512 non-null  float64
     1   latitude                  16512 non-null  float64
     2   housing_median_age        16512 non-null  float64
     3   total_rooms               16512 non-null  float64
     4   total_bedrooms            16354 non-null  float64
     5   population                16512 non-null  float64
     6   households                16512 non-null  float64
     7   median_income             16512 non-null  float64
     8   median_house_value        16512 non-null  float64
     9   ocean_proximity           16512 non-null  object 
     10  room_per_household        16512 non-null  float64
     11  bedrooms_per_room         16354 non-null  float64
     12  population_per_household  16512 non-null  float64
    dtypes: float64(12), object(1)
    memory usage: 1.8+ MB



```python
train_set_temp.values[1]

```




    array([-121.93, 37.05, 14.0, 679.0, 108.0, 306.0, 113.0, 6.4214, 340600.0,
           '<1H OCEAN', 6.008849557522124, 0.15905743740795286,
           2.7079646017699117], dtype=object)




```python
print(list(train_set_temp))
```

    ['longitude', 'latitude', 'housing_median_age', 'total_rooms', 'total_bedrooms', 'population', 'households', 'median_income', 'median_house_value', 'ocean_proximity', 'room_per_household', 'bedrooms_per_room', 'population_per_household']



```python
data_num = train_set_temp.drop("ocean_proximity", axis = 1)
data_num.head()

```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
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
      <th>median_house_value</th>
      <th>room_per_household</th>
      <th>bedrooms_per_room</th>
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
      <td>286600.0</td>
      <td>4.625369</td>
      <td>0.223852</td>
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
      <td>340600.0</td>
      <td>6.008850</td>
      <td>0.159057</td>
      <td>2.707965</td>
    </tr>
    <tr>
      <th>14650</th>
      <td>-117.20</td>
      <td>32.77</td>
      <td>31.0</td>
      <td>1952.0</td>
      <td>471.0</td>
      <td>936.0</td>
      <td>462.0</td>
      <td>2.8621</td>
      <td>196900.0</td>
      <td>4.225108</td>
      <td>0.241291</td>
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
      <td>46300.0</td>
      <td>5.232295</td>
      <td>0.200866</td>
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
      <td>254500.0</td>
      <td>4.505810</td>
      <td>0.231341</td>
      <td>3.047847</td>
    </tr>
  </tbody>
</table>
</div>




```python

```

### Define certain preprocessing, by pipeline


```python
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.base import BaseEstimator, TransformerMixin

rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6
class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room = True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room 
    def fit(self, X, y=None):
        return self # nothing else to do 
    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix] 
        population_per_household = X[:, population_ix] / X[:, households_ix] 
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]
        


num_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy = "median")),
    ("attribs_adder", CombinedAttributesAdder()),
    ("std_scaler", StandardScaler()),
    
])
```


```python

```

### Preprocessing Dataset


```python
strat_train_set_X = strat_train_set.drop("median_house_value", axis=1)
strat_train_set_y = strat_train_set["median_house_value"].copy()

strat_train_set_X_num = strat_train_set_X.drop("ocean_proximity", axis = 1)

num_col  = list(strat_train_set_X_num)
cat_col = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
    ("num", num_pipe, num_col),
    ("cat", OneHotEncoder(), cat_col)
]
)

X_train_pre = full_pipeline.fit_transform(strat_train_set_X)
y_train_pre = strat_train_set_y.copy()


```


```python

```


```python
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV

forest_reg = RandomForestRegressor()
params_grid = [
    {"n_estimators":[3,10,30], "max_features":[2,4,6,8]}
]

grid_search = GridSearchCV(forest_reg, params_grid, cv = 5, scoring = "neg_mean_squared_error", return_train_score = True)
grid_search.fit(X_train_pre, y_train_pre)


```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]}],
                 return_train_score=True, scoring='neg_mean_squared_error')




```python
grid_search.best_params_
```




    {'max_features': 6, 'n_estimators': 30}




```python
grid_search.cv_results_
```




    {'mean_fit_time': array([0.0423553 , 0.11447544, 0.34244328, 0.06016507, 0.19591303,
            0.58611832, 0.084304  , 0.27418165, 0.82677751, 0.10675931,
            0.35628605, 1.07065473]),
     'std_fit_time': array([0.01543273, 0.00166166, 0.00457016, 0.00104785, 0.00245894,
            0.00318926, 0.00075508, 0.00241859, 0.004779  , 0.00078297,
            0.00215536, 0.00358321]),
     'mean_score_time': array([0.00183053, 0.0055541 , 0.01781335, 0.00194063, 0.00591397,
            0.01756878, 0.00188122, 0.00570416, 0.01748796, 0.00173297,
            0.00564075, 0.01761751]),
     'std_score_time': array([2.27634808e-04, 7.72167957e-05, 4.00320246e-04, 1.73083258e-04,
            7.00092114e-05, 7.89931378e-05, 1.12255266e-04, 1.85104782e-04,
            7.65817994e-05, 1.71466558e-05, 6.27878848e-05, 1.17255062e-04]),
     'param_max_features': masked_array(data=[2, 2, 2, 4, 4, 4, 6, 6, 6, 8, 8, 8],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False],
            fill_value='?',
                 dtype=object),
     'param_n_estimators': masked_array(data=[3, 10, 30, 3, 10, 30, 3, 10, 30, 3, 10, 30],
                  mask=[False, False, False, False, False, False, False, False,
                        False, False, False, False],
            fill_value='?',
                 dtype=object),
     'params': [{'max_features': 2, 'n_estimators': 3},
      {'max_features': 2, 'n_estimators': 10},
      {'max_features': 2, 'n_estimators': 30},
      {'max_features': 4, 'n_estimators': 3},
      {'max_features': 4, 'n_estimators': 10},
      {'max_features': 4, 'n_estimators': 30},
      {'max_features': 6, 'n_estimators': 3},
      {'max_features': 6, 'n_estimators': 10},
      {'max_features': 6, 'n_estimators': 30},
      {'max_features': 8, 'n_estimators': 3},
      {'max_features': 8, 'n_estimators': 10},
      {'max_features': 8, 'n_estimators': 30}],
     'split0_test_score': array([-3.89422211e+09, -2.80303871e+09, -2.66087792e+09, -3.41560337e+09,
            -2.69232927e+09, -2.35360687e+09, -3.16687347e+09, -2.68088428e+09,
            -2.36592657e+09, -3.27984190e+09, -2.57759055e+09, -2.39242191e+09]),
     'split1_test_score': array([-4.40073278e+09, -3.29761790e+09, -2.96012290e+09, -3.71537742e+09,
            -2.88032649e+09, -2.61776764e+09, -3.47070172e+09, -2.77372544e+09,
            -2.54484820e+09, -3.30018655e+09, -2.77091074e+09, -2.51200329e+09]),
     'split2_test_score': array([-4.08310821e+09, -3.13799655e+09, -2.86896031e+09, -3.61234803e+09,
            -2.97521629e+09, -2.67080072e+09, -3.68084399e+09, -2.89213467e+09,
            -2.60301370e+09, -3.48977763e+09, -2.80138147e+09, -2.61462474e+09]),
     'split3_test_score': array([-4.14141907e+09, -2.83315226e+09, -2.56557235e+09, -3.46863863e+09,
            -2.65876636e+09, -2.45290504e+09, -3.23640745e+09, -2.56503662e+09,
            -2.35965342e+09, -3.30343589e+09, -2.49895296e+09, -2.35184490e+09]),
     'split4_test_score': array([-4.17651963e+09, -3.37963143e+09, -3.00215758e+09, -3.83487522e+09,
            -2.97497585e+09, -2.70395947e+09, -3.38114346e+09, -2.84430637e+09,
            -2.56713190e+09, -3.23469390e+09, -2.77338705e+09, -2.60007927e+09]),
     'mean_test_score': array([-4.13920036e+09, -3.09028737e+09, -2.81153821e+09, -3.60936854e+09,
            -2.83632285e+09, -2.55980795e+09, -3.38719402e+09, -2.75121748e+09,
            -2.48811476e+09, -3.32158717e+09, -2.68444455e+09, -2.49419482e+09]),
     'std_test_score': array([1.63069023e+08, 2.35632070e+08, 1.70239935e+08, 1.54563820e+08,
            1.36171082e+08, 1.34465980e+08, 1.81393594e+08, 1.17166444e+08,
            1.04015792e+08, 8.75997638e+07, 1.22381634e+08, 1.06445525e+08]),
     'rank_test_score': array([12,  8,  6, 11,  7,  3, 10,  5,  1,  9,  4,  2], dtype=int32),
     'split0_train_score': array([-1.13118721e+09, -5.45065086e+08, -4.31304652e+08, -9.65391800e+08,
            -5.42983512e+08, -3.92789085e+08, -8.99751733e+08, -5.46873533e+08,
            -3.74139893e+08, -9.52686890e+08, -4.91631564e+08, -3.89775427e+08]),
     'split1_train_score': array([-1.12865407e+09, -5.84182490e+08, -4.28470066e+08, -9.74554706e+08,
            -5.00840642e+08, -3.86406237e+08, -9.23107795e+08, -4.88708667e+08,
            -3.78562381e+08, -8.90354274e+08, -4.91801551e+08, -3.74991822e+08]),
     'split2_train_score': array([-1.01351517e+09, -5.40310580e+08, -4.35084751e+08, -9.49130483e+08,
            -5.15146421e+08, -3.82169278e+08, -9.08637617e+08, -4.92822782e+08,
            -3.74361260e+08, -8.86105906e+08, -4.86936413e+08, -3.74491234e+08]),
     'split3_train_score': array([-1.15860154e+09, -5.70947764e+08, -4.35657897e+08, -9.38607067e+08,
            -5.30940725e+08, -4.05300852e+08, -9.37852922e+08, -5.11697006e+08,
            -4.02202115e+08, -8.80327449e+08, -5.21274157e+08, -3.99184818e+08]),
     'split4_train_score': array([-1.12347036e+09, -5.92174986e+08, -4.29267622e+08, -9.53206444e+08,
            -5.14361387e+08, -3.85682770e+08, -8.73762018e+08, -4.93675111e+08,
            -3.71088774e+08, -8.32943160e+08, -4.68975626e+08, -3.92252663e+08]),
     'mean_train_score': array([-1.11108567e+09, -5.66536181e+08, -4.31956998e+08, -9.56178100e+08,
            -5.20854537e+08, -3.90469644e+08, -9.08622417e+08, -5.06755420e+08,
            -3.80070884e+08, -8.88483536e+08, -4.92123862e+08, -3.86139193e+08]),
     'std_train_score': array([50286900.64849573, 20673619.45542805,  2942668.28185422,
            12568474.75932732, 14606315.71751875,  8169039.6897747 ,
            21717206.53081455, 21562840.37803665, 11318319.12059995,
            38164063.19502848, 16809737.7579228 ,  9805344.17104688])}




```python

```
