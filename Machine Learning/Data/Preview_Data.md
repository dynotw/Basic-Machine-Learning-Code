# Data Preview 

**Once we get our data, we can preview data firstly to have a big picture of our data, which is helpful for the following steps.**
**Assuming that our data is stored in pandas Dataframe format, and the following code is based on it.**


# Useful function in Pandas Package
&nbsp;
#### .head()
&emsp;
Let’s take a look at the top five rows using the DataFrame’s ``.head()`` method. We can see the first five samples.

```python
housing = load_housing_data()
housing.head()
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


&nbsp;
#### .info()
&emsp;
The ``.info()`` method is useful to get a quick description of the data, in particular the total number of rows, and each attribute’s type and number of non-null values.
&emsp;
**Notice that the total_bed rooms attribute has only 20,433 non-null values, meaning that 207 districts are miss‐ ing this feature. We will need to take care of this later.**


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


&nbsp;
#### .value_counts()
&emsp;
You can find out what categories exist, and how many districts belong to each category by using the ``.value_counts()`` method:

```python
housing["ocean_proximity"].value_counts()
```



    <1H OCEAN     9136
    INLAND        6551
    NEAR OCEAN    2658
    NEAR BAY      2290
    ISLAND           5
    Name: ocean_proximity, dtype: int64


&nbsp;
#### .describe()
&emsp;
The ``.describe()`` method shows a summary of the numerical attributes. The count, mean, min, and max rows are self-explanatory. Note that the null values are ignored (so, for example, count of total_bedrooms is 20,433, not 20,640)

```python
housing.describe()
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



&nbsp;
#### .hist()
&emsp;
Another quick way to get a feel of the type of data you are dealing with is to plot a histogram for each numerical attribute. A histogram shows the number of instances (on the vertical axis) that have a given value range (on the horizontal axis). You can either plot this one attribute at a time, or you can call the ``.hist()`` method on the whole dataset, and it will plot a histogram for each numerical attribute.


```python
%matplotlib inline
import matplotlib.pyplot as plt
housing.hist(bins=50, figsize=(20,15))
save_fig("attribute_histogram_plots")
plt.show()
```


    
![image](02_end_to_end_machine_learning_project_13_1.png)
    



```python
# to make this notebook's output identical at every run
np.random.seed(42)
```


```python
import numpy as np

# For illustration only. Sklearn has train_test_split()
def split_train_test(data, test_ratio):
    shuffled_indices = np.random.permutation(len(data))
    test_set_size = int(len(data) * test_ratio)
    test_indices = shuffled_indices[:test_set_size]
    train_indices = shuffled_indices[test_set_size:]
    return data.iloc[train_indices], data.iloc[test_indices]
```


```python
train_set, test_set = split_train_test(housing, 0.2)
len(train_set)
```




    16512




```python
len(test_set)
```




    4128




```python
from zlib import crc32

def test_set_check(identifier, test_ratio):
    return crc32(np.int64(identifier)) & 0xffffffff < test_ratio * 2**32

def split_train_test_by_id(data, test_ratio, id_column):
    ids = data[id_column]
    in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio))
    return data.loc[~in_test_set], data.loc[in_test_set]
```

The implementation of `test_set_check()` above works fine in both Python 2 and Python 3. In earlier releases, the following implementation was proposed, which supported any hash function, but was much slower and did not support Python 2:


```python
import hashlib

def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
```

If you want an implementation that supports any hash function and is compatible with both Python 2 and Python 3, here is one:


```python
def test_set_check(identifier, test_ratio, hash=hashlib.md5):
    return bytearray(hash(np.int64(identifier)).digest())[-1] < 256 * test_ratio
```


```python
housing_with_id = housing.reset_index()   # adds an `index` column
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")
```


```python
housing_with_id["id"] = housing["longitude"] * 1000 + housing["latitude"]
train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")
```


```python
test_set.head()
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
      <th>index</th>
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
      <th>id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>8</th>
      <td>8</td>
      <td>-122.26</td>
      <td>37.84</td>
      <td>42.0</td>
      <td>2555.0</td>
      <td>665.0</td>
      <td>1206.0</td>
      <td>595.0</td>
      <td>2.0804</td>
      <td>226700.0</td>
      <td>NEAR BAY</td>
      <td>-122222.16</td>
    </tr>
    <tr>
      <th>10</th>
      <td>10</td>
      <td>-122.26</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>2202.0</td>
      <td>434.0</td>
      <td>910.0</td>
      <td>402.0</td>
      <td>3.2031</td>
      <td>281500.0</td>
      <td>NEAR BAY</td>
      <td>-122222.15</td>
    </tr>
    <tr>
      <th>11</th>
      <td>11</td>
      <td>-122.26</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>3503.0</td>
      <td>752.0</td>
      <td>1504.0</td>
      <td>734.0</td>
      <td>3.2705</td>
      <td>241800.0</td>
      <td>NEAR BAY</td>
      <td>-122222.15</td>
    </tr>
    <tr>
      <th>12</th>
      <td>12</td>
      <td>-122.26</td>
      <td>37.85</td>
      <td>52.0</td>
      <td>2491.0</td>
      <td>474.0</td>
      <td>1098.0</td>
      <td>468.0</td>
      <td>3.0750</td>
      <td>213500.0</td>
      <td>NEAR BAY</td>
      <td>-122222.15</td>
    </tr>
    <tr>
      <th>13</th>
      <td>13</td>
      <td>-122.26</td>
      <td>37.84</td>
      <td>52.0</td>
      <td>696.0</td>
      <td>191.0</td>
      <td>345.0</td>
      <td>174.0</td>
      <td>2.6736</td>
      <td>191300.0</td>
      <td>NEAR BAY</td>
      <td>-122222.16</td>
    </tr>
  </tbody>
</table>
</div>




```python
from sklearn.model_selection import train_test_split

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)
```


```python
test_set.head()
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
      <th>20046</th>
      <td>-119.01</td>
      <td>36.06</td>
      <td>25.0</td>
      <td>1505.0</td>
      <td>NaN</td>
      <td>1392.0</td>
      <td>359.0</td>
      <td>1.6812</td>
      <td>47700.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>3024</th>
      <td>-119.46</td>
      <td>35.14</td>
      <td>30.0</td>
      <td>2943.0</td>
      <td>NaN</td>
      <td>1565.0</td>
      <td>584.0</td>
      <td>2.5313</td>
      <td>45800.0</td>
      <td>INLAND</td>
    </tr>
    <tr>
      <th>15663</th>
      <td>-122.44</td>
      <td>37.80</td>
      <td>52.0</td>
      <td>3830.0</td>
      <td>NaN</td>
      <td>1310.0</td>
      <td>963.0</td>
      <td>3.4801</td>
      <td>500001.0</td>
      <td>NEAR BAY</td>
    </tr>
    <tr>
      <th>20484</th>
      <td>-118.72</td>
      <td>34.28</td>
      <td>17.0</td>
      <td>3051.0</td>
      <td>NaN</td>
      <td>1705.0</td>
      <td>495.0</td>
      <td>5.7376</td>
      <td>218600.0</td>
      <td>&lt;1H OCEAN</td>
    </tr>
    <tr>
      <th>9814</th>
      <td>-121.93</td>
      <td>36.62</td>
      <td>34.0</td>
      <td>2351.0</td>
      <td>NaN</td>
      <td>1063.0</td>
      <td>428.0</td>
      <td>3.7250</td>
      <td>278000.0</td>
      <td>NEAR OCEAN</td>
    </tr>
  </tbody>
</table>
</div>




```python
housing["median_income"].hist()
```




    <AxesSubplot:>




    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_28_1.png)
    



```python
housing["income_cat"] = pd.cut(housing["median_income"],
                               bins=[0., 1.5, 3.0, 4.5, 6., np.inf],
                               labels=[1, 2, 3, 4, 5])
```


```python
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




    <AxesSubplot:>




    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_31_1.png)
    



```python
from sklearn.model_selection import StratifiedShuffleSplit

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in split.split(housing, housing["income_cat"]):
    strat_train_set = housing.loc[train_index]
    strat_test_set = housing.loc[test_index]
```


```python
strat_test_set["income_cat"].value_counts() / len(strat_test_set)
```




    3    0.350533
    2    0.318798
    4    0.176357
    5    0.114583
    1    0.039729
    Name: income_cat, dtype: float64




```python
housing["income_cat"].value_counts() / len(housing)
```




    3    0.350581
    2    0.318847
    4    0.176308
    5    0.114438
    1    0.039826
    Name: income_cat, dtype: float64




```python
def income_cat_proportions(data):
    return data["income_cat"].value_counts() / len(data)

train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

compare_props = pd.DataFrame({
    "Overall": income_cat_proportions(housing),
    "Stratified": income_cat_proportions(strat_test_set),
    "Random": income_cat_proportions(test_set),
}).sort_index()
compare_props["Rand. %error"] = 100 * compare_props["Random"] / compare_props["Overall"] - 100
compare_props["Strat. %error"] = 100 * compare_props["Stratified"] / compare_props["Overall"] - 100
```


```python
compare_props
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
      <th>Overall</th>
      <th>Stratified</th>
      <th>Random</th>
      <th>Rand. %error</th>
      <th>Strat. %error</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>1</th>
      <td>0.039826</td>
      <td>0.039729</td>
      <td>0.040213</td>
      <td>0.973236</td>
      <td>-0.243309</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.318847</td>
      <td>0.318798</td>
      <td>0.324370</td>
      <td>1.732260</td>
      <td>-0.015195</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.350581</td>
      <td>0.350533</td>
      <td>0.358527</td>
      <td>2.266446</td>
      <td>-0.013820</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.176308</td>
      <td>0.176357</td>
      <td>0.167393</td>
      <td>-5.056334</td>
      <td>0.027480</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.114438</td>
      <td>0.114583</td>
      <td>0.109496</td>
      <td>-4.318374</td>
      <td>0.127011</td>
    </tr>
  </tbody>
</table>
</div>




```python
for set_ in (strat_train_set, strat_test_set):
    set_.drop("income_cat", axis=1, inplace=True)
```

# Discover and visualize the data to gain insights


```python
housing = strat_train_set.copy()
```


```python
housing.plot(kind="scatter", x="longitude", y="latitude")
save_fig("bad_visualization_plot")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/3971092627.py in <module>
          1 housing.plot(kind="scatter", x="longitude", y="latitude")
    ----> 2 save_fig("bad_visualization_plot")
    

    NameError: name 'save_fig' is not defined



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_40_1.png)
    



```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
save_fig("better_visualization_plot")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/301336878.py in <module>
          1 housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.1)
    ----> 2 save_fig("better_visualization_plot")
    

    NameError: name 'save_fig' is not defined



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_41_1.png)
    


The argument `sharex=False` fixes a display bug (the x-axis values and legend were not displayed). This is a temporary fix (see: https://github.com/pandas-dev/pandas/issues/10611 ). Thanks to Wilmer Arellano for pointing it out.


```python
housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
             s=housing["population"]/100, label="population", figsize=(10,7),
             c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True,
             sharex=False)
plt.legend()
save_fig("housing_prices_scatterplot")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/952697009.py in <module>
          4              sharex=False)
          5 plt.legend()
    ----> 6 save_fig("housing_prices_scatterplot")
    

    NameError: name 'save_fig' is not defined



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_43_1.png)
    



```python
# Download the California image
images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
os.makedirs(images_path, exist_ok=True)
DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
filename = "california.png"
print("Downloading", filename)
url = DOWNLOAD_ROOT + "images/end_to_end_project/" + filename
urllib.request.urlretrieve(url, os.path.join(images_path, filename))
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/154088554.py in <module>
          1 # Download the California image
    ----> 2 images_path = os.path.join(PROJECT_ROOT_DIR, "images", "end_to_end_project")
          3 os.makedirs(images_path, exist_ok=True)
          4 DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml2/master/"
          5 filename = "california.png"


    NameError: name 'PROJECT_ROOT_DIR' is not defined



```python
import matplotlib.image as mpimg
california_img=mpimg.imread(os.path.join(images_path, filename))
ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
                  s=housing['population']/100, label="Population",
                  c="median_house_value", cmap=plt.get_cmap("jet"),
                  colorbar=False, alpha=0.4)
plt.imshow(california_img, extent=[-124.55, -113.80, 32.45, 42.05], alpha=0.5,
           cmap=plt.get_cmap("jet"))
plt.ylabel("Latitude", fontsize=14)
plt.xlabel("Longitude", fontsize=14)

prices = housing["median_house_value"]
tick_values = np.linspace(prices.min(), prices.max(), 11)
cbar = plt.colorbar(ticks=tick_values/prices.max())
cbar.ax.set_yticklabels(["$%dk"%(round(v/1000)) for v in tick_values], fontsize=14)
cbar.set_label('Median House Value', fontsize=16)

plt.legend(fontsize=16)
save_fig("california_housing_prices_plot")
plt.show()
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/3912764189.py in <module>
          1 import matplotlib.image as mpimg
    ----> 2 california_img=mpimg.imread(os.path.join(images_path, filename))
          3 ax = housing.plot(kind="scatter", x="longitude", y="latitude", figsize=(10,7),
          4                   s=housing['population']/100, label="Population",
          5                   c="median_house_value", cmap=plt.get_cmap("jet"),


    NameError: name 'images_path' is not defined



```python
corr_matrix = housing.corr()
```


```python
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value    1.000000
    median_income         0.687160
    total_rooms           0.135097
    housing_median_age    0.114110
    households            0.064506
    total_bedrooms        0.047689
    population           -0.026920
    longitude            -0.047432
    latitude             -0.142724
    Name: median_house_value, dtype: float64




```python
# from pandas.tools.plotting import scatter_matrix # For older versions of Pandas
from pandas.plotting import scatter_matrix

attributes = ["median_house_value", "median_income", "total_rooms",
              "housing_median_age"]
scatter_matrix(housing[attributes], figsize=(12, 8))
save_fig("scatter_matrix_plot")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/3175067975.py in <module>
          5               "housing_median_age"]
          6 scatter_matrix(housing[attributes], figsize=(12, 8))
    ----> 7 save_fig("scatter_matrix_plot")
    

    NameError: name 'save_fig' is not defined



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_48_1.png)
    



```python
housing.plot(kind="scatter", x="median_income", y="median_house_value",
             alpha=0.1)
plt.axis([0, 16, 0, 550000])
save_fig("income_vs_house_value_scatterplot")
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/1466172184.py in <module>
          2              alpha=0.1)
          3 plt.axis([0, 16, 0, 550000])
    ----> 4 save_fig("income_vs_house_value_scatterplot")
    

    NameError: name 'save_fig' is not defined



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_49_1.png)
    



```python
housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
housing["population_per_household"]=housing["population"]/housing["households"]
```


```python
corr_matrix = housing.corr()
corr_matrix["median_house_value"].sort_values(ascending=False)
```




    median_house_value          1.000000
    median_income               0.687160
    rooms_per_household         0.146285
    total_rooms                 0.135097
    housing_median_age          0.114110
    households                  0.064506
    total_bedrooms              0.047689
    population_per_household   -0.021985
    population                 -0.026920
    longitude                  -0.047432
    latitude                   -0.142724
    bedrooms_per_room          -0.259984
    Name: median_house_value, dtype: float64




```python
housing.plot(kind="scatter", x="rooms_per_household", y="median_house_value",
             alpha=0.2)
plt.axis([0, 5, 0, 520000])
plt.show()
```


    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_52_0.png)
    



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
      <th>rooms_per_household</th>
      <th>bedrooms_per_room</th>
      <th>population_per_household</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16512.000000</td>
      <td>16354.000000</td>
      <td>16512.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>-119.575834</td>
      <td>35.639577</td>
      <td>28.653101</td>
      <td>2622.728319</td>
      <td>534.973890</td>
      <td>1419.790819</td>
      <td>497.060380</td>
      <td>3.875589</td>
      <td>206990.920724</td>
      <td>5.440341</td>
      <td>0.212878</td>
      <td>3.096437</td>
    </tr>
    <tr>
      <th>std</th>
      <td>2.001860</td>
      <td>2.138058</td>
      <td>12.574726</td>
      <td>2138.458419</td>
      <td>412.699041</td>
      <td>1115.686241</td>
      <td>375.720845</td>
      <td>1.904950</td>
      <td>115703.014830</td>
      <td>2.611712</td>
      <td>0.057379</td>
      <td>11.584826</td>
    </tr>
    <tr>
      <th>min</th>
      <td>-124.350000</td>
      <td>32.540000</td>
      <td>1.000000</td>
      <td>6.000000</td>
      <td>2.000000</td>
      <td>3.000000</td>
      <td>2.000000</td>
      <td>0.499900</td>
      <td>14999.000000</td>
      <td>1.130435</td>
      <td>0.100000</td>
      <td>0.692308</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>-121.800000</td>
      <td>33.940000</td>
      <td>18.000000</td>
      <td>1443.000000</td>
      <td>295.000000</td>
      <td>784.000000</td>
      <td>279.000000</td>
      <td>2.566775</td>
      <td>119800.000000</td>
      <td>4.442040</td>
      <td>0.175304</td>
      <td>2.431287</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>-118.510000</td>
      <td>34.260000</td>
      <td>29.000000</td>
      <td>2119.500000</td>
      <td>433.000000</td>
      <td>1164.000000</td>
      <td>408.000000</td>
      <td>3.540900</td>
      <td>179500.000000</td>
      <td>5.232284</td>
      <td>0.203031</td>
      <td>2.817653</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>-118.010000</td>
      <td>37.720000</td>
      <td>37.000000</td>
      <td>3141.000000</td>
      <td>644.000000</td>
      <td>1719.250000</td>
      <td>602.000000</td>
      <td>4.744475</td>
      <td>263900.000000</td>
      <td>6.056361</td>
      <td>0.239831</td>
      <td>3.281420</td>
    </tr>
    <tr>
      <th>max</th>
      <td>-114.310000</td>
      <td>41.950000</td>
      <td>52.000000</td>
      <td>39320.000000</td>
      <td>6210.000000</td>
      <td>35682.000000</td>
      <td>5358.000000</td>
      <td>15.000100</td>
      <td>500001.000000</td>
      <td>141.909091</td>
      <td>1.000000</td>
      <td>1243.333333</td>
    </tr>
  </tbody>
</table>
</div>



# Prepare the data for Machine Learning algorithms


```python
housing = strat_train_set.drop("median_house_value", axis=1) # drop labels for training set
housing_labels = strat_train_set["median_house_value"].copy()
```


```python
sample_incomplete_rows = housing[housing.isnull().any(axis=1)].head()
sample_incomplete_rows
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




```python
sample_incomplete_rows.dropna(subset=["total_bedrooms"])    # option 1
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
      <th>ocean_proximity</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>




```python
sample_incomplete_rows.drop("total_bedrooms", axis=1)       # option 2
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




```python
median = housing["total_bedrooms"].median()
sample_incomplete_rows["total_bedrooms"].fillna(median, inplace=True) # option 3
```


```python
sample_incomplete_rows
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




```python
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy="median")
```

Remove the text attribute because median can only be calculated on numerical attributes:


```python
housing_num = housing.drop("ocean_proximity", axis=1)
# alternatively: housing_num = housing.select_dtypes(include=[np.number])
```


```python
imputer.fit(housing_num)
```




    SimpleImputer(strategy='median')




```python
imputer.statistics_
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])



Check that this is the same as manually computing the median of each attribute:


```python
housing_num.median().values
```




    array([-118.51  ,   34.26  ,   29.    , 2119.5   ,  433.    , 1164.    ,
            408.    ,    3.5409])



Transform the training set:


```python
X = imputer.transform(housing_num)
```


```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing.index)
```


```python
housing_tr.loc[sample_incomplete_rows.index.values]
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




```python
imputer.strategy
```




    'median'




```python
housing_tr = pd.DataFrame(X, columns=housing_num.columns,
                          index=housing_num.index)
```


```python
housing_tr.head()
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
    </tr>
  </tbody>
</table>
</div>



Now let's preprocess the categorical input feature, `ocean_proximity`:


```python
housing_cat = housing[["ocean_proximity"]]
housing_cat.head(10)
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




```python
ordinal_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]




```python
from sklearn.preprocessing import OneHotEncoder

cat_encoder = OneHotEncoder()
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    <16512x5 sparse matrix of type '<class 'numpy.float64'>'
    	with 16512 stored elements in Compressed Sparse Row format>



By default, the `OneHotEncoder` class returns a sparse array, but we can convert it to a dense array if needed by calling the `toarray()` method:


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



Alternatively, you can set `sparse=False` when creating the `OneHotEncoder`:


```python
cat_encoder = OneHotEncoder(sparse=False)
housing_cat_1hot = cat_encoder.fit_transform(housing_cat)
housing_cat_1hot
```




    array([[1., 0., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 0., 1.],
           ...,
           [0., 1., 0., 0., 0.],
           [1., 0., 0., 0., 0.],
           [0., 0., 0., 1., 0.]])




```python
cat_encoder.categories_
```




    [array(['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'],
           dtype=object)]



Let's create a custom transformer to add extra attributes:


```python
from sklearn.base import BaseEstimator, TransformerMixin

# column index
rooms_ix, bedrooms_ix, population_ix, households_ix = 3, 4, 5, 6

class CombinedAttributesAdder(BaseEstimator, TransformerMixin):
    def __init__(self, add_bedrooms_per_room=True): # no *args or **kargs
        self.add_bedrooms_per_room = add_bedrooms_per_room
    def fit(self, X, y=None):
        return self  # nothing else to do
    def transform(self, X):
        rooms_per_household = X[:, rooms_ix] / X[:, households_ix]
        population_per_household = X[:, population_ix] / X[:, households_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household,
                         bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]

attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
housing_extra_attribs = attr_adder.transform(housing.values)
```

Note that I hard coded the indices (3, 4, 5, 6) for concision and clarity in the book, but it would be much cleaner to get them dynamically, like this:


```python
col_names = "total_rooms", "total_bedrooms", "population", "households"
rooms_ix, bedrooms_ix, population_ix, households_ix = [
    housing.columns.get_loc(c) for c in col_names] # get the column indices
```

Also, `housing_extra_attribs` is a NumPy array, we've lost the column names (unfortunately, that's a problem with Scikit-Learn). To recover a `DataFrame`, you could run this:


```python
housing_extra_attribs = pd.DataFrame(
    housing_extra_attribs,
    columns=list(housing.columns)+["rooms_per_household", "population_per_household"],
    index=housing.index)
housing_extra_attribs.head()
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



Now let's build a pipeline for preprocessing the numerical attributes:


```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

housing_num_tr = num_pipeline.fit_transform(housing_num)
```


```python
housing_num_tr
```




    array([[-1.15604281,  0.77194962,  0.74333089, ..., -0.31205452,
            -0.08649871,  0.15531753],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.21768338,
            -0.03353391, -0.83628902],
           [ 1.18684903, -1.34218285,  0.18664186, ..., -0.46531516,
            -0.09240499,  0.4222004 ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.3469342 ,
            -0.03055414, -0.52177644],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.02499488,
             0.06150916, -0.30340741],
           [-1.43579109,  0.99645926,  1.85670895, ..., -0.22852947,
            -0.09586294,  0.10180567]])




```python
from sklearn.compose import ColumnTransformer

num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_attribs),
        ("cat", OneHotEncoder(), cat_attribs),
    ])

housing_prepared = full_pipeline.fit_transform(housing)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/2987397867.py in <module>
          1 from sklearn.compose import ColumnTransformer
          2 
    ----> 3 num_attribs = list(housing_num)
          4 cat_attribs = ["ocean_proximity"]
          5 


    NameError: name 'housing_num' is not defined



```python
housing_prepared
```




    array([[-1.15604281,  0.77194962,  0.74333089, ...,  0.        ,
             0.        ,  0.        ],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.        ,
             0.        ,  0.        ],
           [ 1.18684903, -1.34218285,  0.18664186, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.        ,
             0.        ,  0.        ],
           [-1.43579109,  0.99645926,  1.85670895, ...,  0.        ,
             1.        ,  0.        ]])




```python
housing_prepared.shape
```




    (16512, 16)



For reference, here is the old solution based on a `DataFrameSelector` transformer (to just select a subset of the Pandas `DataFrame` columns), and a `FeatureUnion`:


```python
from sklearn.base import BaseEstimator, TransformerMixin

# Create a class to select numerical or categorical columns 
class OldDataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X[self.attribute_names].values
```

Now let's join all these components into a big pipeline that will preprocess both the numerical and the categorical features:


```python
num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

old_num_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(num_attribs)),
        ('imputer', SimpleImputer(strategy="median")),
        ('attribs_adder', CombinedAttributesAdder()),
        ('std_scaler', StandardScaler()),
    ])

old_cat_pipeline = Pipeline([
        ('selector', OldDataFrameSelector(cat_attribs)),
        ('cat_encoder', OneHotEncoder(sparse=False)),
    ])
```


```python
from sklearn.pipeline import FeatureUnion

old_full_pipeline = FeatureUnion(transformer_list=[
        ("num_pipeline", old_num_pipeline),
        ("cat_pipeline", old_cat_pipeline),
    ])
```


```python
old_housing_prepared = old_full_pipeline.fit_transform(housing)
old_housing_prepared
```




    array([[-1.15604281,  0.77194962,  0.74333089, ...,  0.        ,
             0.        ,  0.        ],
           [-1.17602483,  0.6596948 , -1.1653172 , ...,  0.        ,
             0.        ,  0.        ],
           [ 1.18684903, -1.34218285,  0.18664186, ...,  0.        ,
             0.        ,  1.        ],
           ...,
           [ 1.58648943, -0.72478134, -1.56295222, ...,  0.        ,
             0.        ,  0.        ],
           [ 0.78221312, -0.85106801,  0.18664186, ...,  0.        ,
             0.        ,  0.        ],
           [-1.43579109,  0.99645926,  1.85670895, ...,  0.        ,
             1.        ,  0.        ]])



The result is the same as with the `ColumnTransformer`:


```python
np.allclose(housing_prepared, old_housing_prepared)
```




    True



# Select and train a model 


```python
from sklearn.linear_model import LinearRegression

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
```




    LinearRegression()




```python
# let's try the full preprocessing pipeline on a few training instances
some_data = housing.iloc[:5]
some_labels = housing_labels.iloc[:5]
some_data_prepared = full_pipeline.transform(some_data)

print("Predictions:", lin_reg.predict(some_data_prepared))
```

    Predictions: [210644.60459286 317768.80697211 210956.43331178  59218.98886849
     189747.55849879]


Compare against the actual values:


```python
print("Labels:", list(some_labels))
```

    Labels: [286600.0, 340600.0, 196900.0, 46300.0, 254500.0]



```python
some_data_prepared
```




    array([[-1.15604281,  0.77194962,  0.74333089, -0.49323393, -0.44543821,
            -0.63621141, -0.42069842, -0.61493744, -0.31205452, -0.08649871,
             0.15531753,  1.        ,  0.        ,  0.        ,  0.        ,
             0.        ],
           [-1.17602483,  0.6596948 , -1.1653172 , -0.90896655, -1.0369278 ,
            -0.99833135, -1.02222705,  1.33645936,  0.21768338, -0.03353391,
            -0.83628902,  1.        ,  0.        ,  0.        ,  0.        ,
             0.        ],
           [ 1.18684903, -1.34218285,  0.18664186, -0.31365989, -0.15334458,
            -0.43363936, -0.0933178 , -0.5320456 , -0.46531516, -0.09240499,
             0.4222004 ,  0.        ,  0.        ,  0.        ,  0.        ,
             1.        ],
           [-0.01706767,  0.31357576, -0.29052016, -0.36276217, -0.39675594,
             0.03604096, -0.38343559, -1.04556555, -0.07966124,  0.08973561,
            -0.19645314,  0.        ,  1.        ,  0.        ,  0.        ,
             0.        ],
           [ 0.49247384, -0.65929936, -0.92673619,  1.85619316,  2.41221109,
             2.72415407,  2.57097492, -0.44143679, -0.35783383, -0.00419445,
             0.2699277 ,  1.        ,  0.        ,  0.        ,  0.        ,
             0.        ]])




```python
from sklearn.metrics import mean_squared_error

housing_predictions = lin_reg.predict(housing_prepared)
lin_mse = mean_squared_error(housing_labels, housing_predictions)
lin_rmse = np.sqrt(lin_mse)
lin_rmse
```




    68628.19819848923



**Note**: since Scikit-Learn 0.22, you can get the RMSE directly by calling the `mean_squared_error()` function with `squared=False`.


```python
from sklearn.metrics import mean_absolute_error

lin_mae = mean_absolute_error(housing_labels, housing_predictions)
lin_mae
```




    49439.89599001897




```python
from sklearn.tree import DecisionTreeRegressor

tree_reg = DecisionTreeRegressor(random_state=42)
tree_reg.fit(housing_prepared, housing_labels)
```




    DecisionTreeRegressor(random_state=42)




```python
housing_predictions = tree_reg.predict(housing_prepared)
tree_mse = mean_squared_error(housing_labels, housing_predictions)
tree_rmse = np.sqrt(tree_mse)
tree_rmse
```




    0.0



# Fine-tune your model


```python
from sklearn.model_selection import cross_val_score

scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-scores)
```


```python
def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())

display_scores(tree_rmse_scores)
```

    Scores: [70194.33680785 66855.16363941 72432.58244769 70758.73896782
     71115.88230639 75585.14172901 70262.86139133 70273.6325285
     75366.87952553 71231.65726027]
    Mean: 71407.68766037929
    Standard deviation: 2439.4345041191004



```python
lin_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                             scoring="neg_mean_squared_error", cv=10)
lin_rmse_scores = np.sqrt(-lin_scores)
display_scores(lin_rmse_scores)
```

    Scores: [66782.73843989 66960.118071   70347.95244419 74739.57052552
     68031.13388938 71193.84183426 64969.63056405 68281.61137997
     71552.91566558 67665.10082067]
    Mean: 69052.46136345083
    Standard deviation: 2731.674001798342


**Note**: we specify `n_estimators=100` to be future-proof since the default value is going to change to 100 in Scikit-Learn 0.22 (for simplicity, this is not shown in the book).


```python
from sklearn.ensemble import RandomForestRegressor

forest_reg = RandomForestRegressor(n_estimators=100, random_state=42)
forest_reg.fit(housing_prepared, housing_labels)
```




    RandomForestRegressor(random_state=42)




```python
housing_predictions = forest_reg.predict(housing_prepared)
forest_mse = mean_squared_error(housing_labels, housing_predictions)
forest_rmse = np.sqrt(forest_mse)
forest_rmse
```




    18603.515021376355




```python
from sklearn.model_selection import cross_val_score

forest_scores = cross_val_score(forest_reg, housing_prepared, housing_labels,
                                scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)
display_scores(forest_rmse_scores)
```

    Scores: [49519.80364233 47461.9115823  50029.02762854 52325.28068953
     49308.39426421 53446.37892622 48634.8036574  47585.73832311
     53490.10699751 50021.5852922 ]
    Mean: 50182.303100336096
    Standard deviation: 2097.0810550985693



```python
scores = cross_val_score(lin_reg, housing_prepared, housing_labels, scoring="neg_mean_squared_error", cv=10)
pd.Series(np.sqrt(-scores)).describe()
```




    count       10.000000
    mean     69052.461363
    std       2879.437224
    min      64969.630564
    25%      67136.363758
    50%      68156.372635
    75%      70982.369487
    max      74739.570526
    dtype: float64




```python
from sklearn.svm import SVR

svm_reg = SVR(kernel="linear")
svm_reg.fit(housing_prepared, housing_labels)
housing_predictions = svm_reg.predict(housing_prepared)
svm_mse = mean_squared_error(housing_labels, housing_predictions)
svm_rmse = np.sqrt(svm_mse)
svm_rmse
```




    111094.6308539982




```python
from sklearn.model_selection import GridSearchCV

param_grid = [
    # try 12 (3×4) combinations of hyperparameters
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    # then try 6 (2×3) combinations with bootstrap set as False
    {'bootstrap': [False], 'n_estimators': [3, 10], 'max_features': [2, 3, 4]},
  ]

forest_reg = RandomForestRegressor(random_state=42)
# train across 5 folds, that's a total of (12+6)*5=90 rounds of training 
grid_search = GridSearchCV(forest_reg, param_grid, cv=5,
                           scoring='neg_mean_squared_error',
                           return_train_score=True)
grid_search.fit(housing_prepared, housing_labels)
```




    GridSearchCV(cv=5, estimator=RandomForestRegressor(random_state=42),
                 param_grid=[{'max_features': [2, 4, 6, 8],
                              'n_estimators': [3, 10, 30]},
                             {'bootstrap': [False], 'max_features': [2, 3, 4],
                              'n_estimators': [3, 10]}],
                 return_train_score=True, scoring='neg_mean_squared_error')



The best hyperparameter combination found:


```python
grid_search.best_params_
```




    {'max_features': 8, 'n_estimators': 30}




```python
grid_search.best_estimator_
```




    RandomForestRegressor(max_features=8, n_estimators=30, random_state=42)



Let's look at the score of each hyperparameter combination tested during the grid search:


```python
cvres = grid_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    63669.11631261028 {'max_features': 2, 'n_estimators': 3}
    55627.099719926795 {'max_features': 2, 'n_estimators': 10}
    53384.57275149205 {'max_features': 2, 'n_estimators': 30}
    60965.950449450494 {'max_features': 4, 'n_estimators': 3}
    52741.04704299915 {'max_features': 4, 'n_estimators': 10}
    50377.40461678399 {'max_features': 4, 'n_estimators': 30}
    58663.93866579625 {'max_features': 6, 'n_estimators': 3}
    52006.19873526564 {'max_features': 6, 'n_estimators': 10}
    50146.51167415009 {'max_features': 6, 'n_estimators': 30}
    57869.25276169646 {'max_features': 8, 'n_estimators': 3}
    51711.127883959234 {'max_features': 8, 'n_estimators': 10}
    49682.273345071546 {'max_features': 8, 'n_estimators': 30}
    62895.06951262424 {'bootstrap': False, 'max_features': 2, 'n_estimators': 3}
    54658.176157539405 {'bootstrap': False, 'max_features': 2, 'n_estimators': 10}
    59470.40652318466 {'bootstrap': False, 'max_features': 3, 'n_estimators': 3}
    52724.9822587892 {'bootstrap': False, 'max_features': 3, 'n_estimators': 10}
    57490.5691951261 {'bootstrap': False, 'max_features': 4, 'n_estimators': 3}
    51009.495668875716 {'bootstrap': False, 'max_features': 4, 'n_estimators': 10}



```python
pd.DataFrame(grid_search.cv_results_)
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
      <th>mean_fit_time</th>
      <th>std_fit_time</th>
      <th>mean_score_time</th>
      <th>std_score_time</th>
      <th>param_max_features</th>
      <th>param_n_estimators</th>
      <th>param_bootstrap</th>
      <th>params</th>
      <th>split0_test_score</th>
      <th>split1_test_score</th>
      <th>...</th>
      <th>mean_test_score</th>
      <th>std_test_score</th>
      <th>rank_test_score</th>
      <th>split0_train_score</th>
      <th>split1_train_score</th>
      <th>split2_train_score</th>
      <th>split3_train_score</th>
      <th>split4_train_score</th>
      <th>mean_train_score</th>
      <th>std_train_score</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.050905</td>
      <td>0.004097</td>
      <td>0.002766</td>
      <td>0.000256</td>
      <td>2</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 3}</td>
      <td>-3.837622e+09</td>
      <td>-4.147108e+09</td>
      <td>...</td>
      <td>-4.053756e+09</td>
      <td>1.519591e+08</td>
      <td>18</td>
      <td>-1.064113e+09</td>
      <td>-1.105142e+09</td>
      <td>-1.116550e+09</td>
      <td>-1.112342e+09</td>
      <td>-1.129650e+09</td>
      <td>-1.105559e+09</td>
      <td>2.220402e+07</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0.143706</td>
      <td>0.002170</td>
      <td>0.007205</td>
      <td>0.000304</td>
      <td>2</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 10}</td>
      <td>-3.047771e+09</td>
      <td>-3.254861e+09</td>
      <td>...</td>
      <td>-3.094374e+09</td>
      <td>1.327062e+08</td>
      <td>11</td>
      <td>-5.927175e+08</td>
      <td>-5.870952e+08</td>
      <td>-5.776964e+08</td>
      <td>-5.716332e+08</td>
      <td>-5.802501e+08</td>
      <td>-5.818785e+08</td>
      <td>7.345821e+06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0.410306</td>
      <td>0.004403</td>
      <td>0.019903</td>
      <td>0.000964</td>
      <td>2</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 2, 'n_estimators': 30}</td>
      <td>-2.689185e+09</td>
      <td>-3.021086e+09</td>
      <td>...</td>
      <td>-2.849913e+09</td>
      <td>1.626875e+08</td>
      <td>9</td>
      <td>-4.381089e+08</td>
      <td>-4.391272e+08</td>
      <td>-4.371702e+08</td>
      <td>-4.376955e+08</td>
      <td>-4.452654e+08</td>
      <td>-4.394734e+08</td>
      <td>2.966320e+06</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0.069762</td>
      <td>0.000987</td>
      <td>0.002409</td>
      <td>0.000080</td>
      <td>4</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 3}</td>
      <td>-3.730181e+09</td>
      <td>-3.786886e+09</td>
      <td>...</td>
      <td>-3.716847e+09</td>
      <td>1.631510e+08</td>
      <td>16</td>
      <td>-9.865163e+08</td>
      <td>-1.012565e+09</td>
      <td>-9.169425e+08</td>
      <td>-1.037400e+09</td>
      <td>-9.707739e+08</td>
      <td>-9.848396e+08</td>
      <td>4.084607e+07</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0.227188</td>
      <td>0.001444</td>
      <td>0.006829</td>
      <td>0.000090</td>
      <td>4</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 10}</td>
      <td>-2.666283e+09</td>
      <td>-2.784511e+09</td>
      <td>...</td>
      <td>-2.781618e+09</td>
      <td>1.268607e+08</td>
      <td>8</td>
      <td>-5.097115e+08</td>
      <td>-5.162820e+08</td>
      <td>-4.962893e+08</td>
      <td>-5.436192e+08</td>
      <td>-5.160297e+08</td>
      <td>-5.163863e+08</td>
      <td>1.542862e+07</td>
    </tr>
    <tr>
      <th>5</th>
      <td>0.711381</td>
      <td>0.038618</td>
      <td>0.020686</td>
      <td>0.001864</td>
      <td>4</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 4, 'n_estimators': 30}</td>
      <td>-2.387153e+09</td>
      <td>-2.588448e+09</td>
      <td>...</td>
      <td>-2.537883e+09</td>
      <td>1.214614e+08</td>
      <td>3</td>
      <td>-3.838835e+08</td>
      <td>-3.880268e+08</td>
      <td>-3.790867e+08</td>
      <td>-4.040957e+08</td>
      <td>-3.845520e+08</td>
      <td>-3.879289e+08</td>
      <td>8.571233e+06</td>
    </tr>
    <tr>
      <th>6</th>
      <td>0.094580</td>
      <td>0.003861</td>
      <td>0.002426</td>
      <td>0.000118</td>
      <td>6</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 3}</td>
      <td>-3.119657e+09</td>
      <td>-3.586319e+09</td>
      <td>...</td>
      <td>-3.441458e+09</td>
      <td>1.893056e+08</td>
      <td>14</td>
      <td>-9.245343e+08</td>
      <td>-8.886939e+08</td>
      <td>-9.353135e+08</td>
      <td>-9.009801e+08</td>
      <td>-8.624664e+08</td>
      <td>-9.023976e+08</td>
      <td>2.591445e+07</td>
    </tr>
    <tr>
      <th>7</th>
      <td>0.311034</td>
      <td>0.005247</td>
      <td>0.006980</td>
      <td>0.000283</td>
      <td>6</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 10}</td>
      <td>-2.549663e+09</td>
      <td>-2.782039e+09</td>
      <td>...</td>
      <td>-2.704645e+09</td>
      <td>1.471569e+08</td>
      <td>6</td>
      <td>-4.980344e+08</td>
      <td>-5.045869e+08</td>
      <td>-4.994664e+08</td>
      <td>-4.990325e+08</td>
      <td>-5.055542e+08</td>
      <td>-5.013349e+08</td>
      <td>3.100456e+06</td>
    </tr>
    <tr>
      <th>8</th>
      <td>0.979656</td>
      <td>0.048790</td>
      <td>0.021028</td>
      <td>0.001812</td>
      <td>6</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 6, 'n_estimators': 30}</td>
      <td>-2.370010e+09</td>
      <td>-2.583638e+09</td>
      <td>...</td>
      <td>-2.514673e+09</td>
      <td>1.285080e+08</td>
      <td>2</td>
      <td>-3.838538e+08</td>
      <td>-3.804711e+08</td>
      <td>-3.805218e+08</td>
      <td>-3.856095e+08</td>
      <td>-3.901917e+08</td>
      <td>-3.841296e+08</td>
      <td>3.617057e+06</td>
    </tr>
    <tr>
      <th>9</th>
      <td>0.118484</td>
      <td>0.001009</td>
      <td>0.002239</td>
      <td>0.000068</td>
      <td>8</td>
      <td>3</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 3}</td>
      <td>-3.353504e+09</td>
      <td>-3.348552e+09</td>
      <td>...</td>
      <td>-3.348850e+09</td>
      <td>1.241939e+08</td>
      <td>13</td>
      <td>-9.228123e+08</td>
      <td>-8.553031e+08</td>
      <td>-8.603321e+08</td>
      <td>-8.881964e+08</td>
      <td>-9.151287e+08</td>
      <td>-8.883545e+08</td>
      <td>2.750227e+07</td>
    </tr>
    <tr>
      <th>10</th>
      <td>0.401726</td>
      <td>0.005465</td>
      <td>0.007028</td>
      <td>0.000345</td>
      <td>8</td>
      <td>10</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 10}</td>
      <td>-2.571970e+09</td>
      <td>-2.718994e+09</td>
      <td>...</td>
      <td>-2.674041e+09</td>
      <td>1.392777e+08</td>
      <td>5</td>
      <td>-4.932416e+08</td>
      <td>-4.815238e+08</td>
      <td>-4.730979e+08</td>
      <td>-5.155367e+08</td>
      <td>-4.985555e+08</td>
      <td>-4.923911e+08</td>
      <td>1.459294e+07</td>
    </tr>
    <tr>
      <th>11</th>
      <td>1.236572</td>
      <td>0.036875</td>
      <td>0.019325</td>
      <td>0.000252</td>
      <td>8</td>
      <td>30</td>
      <td>NaN</td>
      <td>{'max_features': 8, 'n_estimators': 30}</td>
      <td>-2.357390e+09</td>
      <td>-2.546640e+09</td>
      <td>...</td>
      <td>-2.468328e+09</td>
      <td>1.091662e+08</td>
      <td>1</td>
      <td>-3.841658e+08</td>
      <td>-3.744500e+08</td>
      <td>-3.773239e+08</td>
      <td>-3.882250e+08</td>
      <td>-3.810005e+08</td>
      <td>-3.810330e+08</td>
      <td>4.871017e+06</td>
    </tr>
    <tr>
      <th>12</th>
      <td>0.064666</td>
      <td>0.001042</td>
      <td>0.002780</td>
      <td>0.000240</td>
      <td>2</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-3.785816e+09</td>
      <td>-4.166012e+09</td>
      <td>...</td>
      <td>-3.955790e+09</td>
      <td>1.900964e+08</td>
      <td>17</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>13</th>
      <td>0.213941</td>
      <td>0.000996</td>
      <td>0.007956</td>
      <td>0.000267</td>
      <td>2</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 2, 'n_est...</td>
      <td>-2.810721e+09</td>
      <td>-3.107789e+09</td>
      <td>...</td>
      <td>-2.987516e+09</td>
      <td>1.539234e+08</td>
      <td>10</td>
      <td>-6.056477e-02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-2.967449e+00</td>
      <td>-6.056027e-01</td>
      <td>1.181156e+00</td>
    </tr>
    <tr>
      <th>14</th>
      <td>0.090480</td>
      <td>0.003167</td>
      <td>0.002681</td>
      <td>0.000082</td>
      <td>3</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-3.618324e+09</td>
      <td>-3.441527e+09</td>
      <td>...</td>
      <td>-3.536729e+09</td>
      <td>7.795057e+07</td>
      <td>15</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-6.072840e+01</td>
      <td>-1.214568e+01</td>
      <td>2.429136e+01</td>
    </tr>
    <tr>
      <th>15</th>
      <td>0.286396</td>
      <td>0.004578</td>
      <td>0.008019</td>
      <td>0.000384</td>
      <td>3</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 3, 'n_est...</td>
      <td>-2.757999e+09</td>
      <td>-2.851737e+09</td>
      <td>...</td>
      <td>-2.779924e+09</td>
      <td>6.286720e+07</td>
      <td>7</td>
      <td>-2.089484e+01</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-5.465556e+00</td>
      <td>-5.272080e+00</td>
      <td>8.093117e+00</td>
    </tr>
    <tr>
      <th>16</th>
      <td>0.109239</td>
      <td>0.002999</td>
      <td>0.003399</td>
      <td>0.001579</td>
      <td>4</td>
      <td>3</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-3.134040e+09</td>
      <td>-3.559375e+09</td>
      <td>...</td>
      <td>-3.305166e+09</td>
      <td>1.879165e+08</td>
      <td>12</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>0.000000e+00</td>
      <td>0.000000e+00</td>
    </tr>
    <tr>
      <th>17</th>
      <td>0.370459</td>
      <td>0.017424</td>
      <td>0.007863</td>
      <td>0.000056</td>
      <td>4</td>
      <td>10</td>
      <td>False</td>
      <td>{'bootstrap': False, 'max_features': 4, 'n_est...</td>
      <td>-2.525578e+09</td>
      <td>-2.710011e+09</td>
      <td>...</td>
      <td>-2.601969e+09</td>
      <td>1.088048e+08</td>
      <td>4</td>
      <td>-0.000000e+00</td>
      <td>-1.514119e-02</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-0.000000e+00</td>
      <td>-3.028238e-03</td>
      <td>6.056477e-03</td>
    </tr>
  </tbody>
</table>
<p>18 rows × 23 columns</p>
</div>




```python
from sklearn.model_selection import RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint

param_distribs = {
        'n_estimators': randint(low=1, high=200),
        'max_features': randint(low=1, high=8),
    }

forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/802026403.py in <module>
         11 rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
         12                                 n_iter=10, cv=5, scoring='neg_mean_squared_error', random_state=42)
    ---> 13 rnd_search.fit(housing_prepared, housing_labels)
    

    NameError: name 'housing_prepared' is not defined



```python
cvres = rnd_search.cv_results_
for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
    print(np.sqrt(-mean_score), params)
```

    49150.70756927707 {'max_features': 7, 'n_estimators': 180}
    51389.889203389284 {'max_features': 5, 'n_estimators': 15}
    50796.155224308866 {'max_features': 3, 'n_estimators': 72}
    50835.13360315349 {'max_features': 5, 'n_estimators': 21}
    49280.9449827171 {'max_features': 7, 'n_estimators': 122}
    50774.90662363929 {'max_features': 3, 'n_estimators': 75}
    50682.78888164288 {'max_features': 3, 'n_estimators': 88}
    49608.99608105296 {'max_features': 5, 'n_estimators': 100}
    50473.61930350219 {'max_features': 3, 'n_estimators': 150}
    64429.84143294435 {'max_features': 5, 'n_estimators': 2}



```python
feature_importances = grid_search.best_estimator_.feature_importances_
feature_importances
```




    array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
           1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
           5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
           1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03])




```python
extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
#cat_encoder = cat_pipeline.named_steps["cat_encoder"] # old solution
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
sorted(zip(feature_importances, attributes), reverse=True)
```




    [(0.36615898061813423, 'median_income'),
     (0.16478099356159054, 'INLAND'),
     (0.10879295677551575, 'pop_per_hhold'),
     (0.07334423551601243, 'longitude'),
     (0.06290907048262032, 'latitude'),
     (0.056419179181954014, 'rooms_per_hhold'),
     (0.053351077347675815, 'bedrooms_per_room'),
     (0.04114379847872964, 'housing_median_age'),
     (0.014874280890402769, 'population'),
     (0.014672685420543239, 'total_rooms'),
     (0.014257599323407808, 'households'),
     (0.014106483453584104, 'total_bedrooms'),
     (0.010311488326303788, '<1H OCEAN'),
     (0.0028564746373201584, 'NEAR OCEAN'),
     (0.0019604155994780706, 'NEAR BAY'),
     (6.0280386727366e-05, 'ISLAND')]




```python
final_model = grid_search.best_estimator_

X_test = strat_test_set.drop("median_house_value", axis=1)
y_test = strat_test_set["median_house_value"].copy()

X_test_prepared = full_pipeline.transform(X_test)
final_predictions = final_model.predict(X_test_prepared)

final_mse = mean_squared_error(y_test, final_predictions)
final_rmse = np.sqrt(final_mse)
```


```python
final_rmse
```




    47730.22690385927



We can compute a 95% confidence interval for the test RMSE:


```python
from scipy import stats

confidence = 0.95
squared_errors = (final_predictions - y_test) ** 2
np.sqrt(stats.t.interval(confidence, len(squared_errors) - 1,
                         loc=squared_errors.mean(),
                         scale=stats.sem(squared_errors)))
```




    array([45685.10470776, 49691.25001878])



We could compute the interval manually like this:


```python
m = len(squared_errors)
mean = squared_errors.mean()
tscore = stats.t.ppf((1 + confidence) / 2, df=m - 1)
tmargin = tscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - tmargin), np.sqrt(mean + tmargin)
```




    (45685.10470776014, 49691.25001877871)



Alternatively, we could use a z-scores rather than t-scores:


```python
zscore = stats.norm.ppf((1 + confidence) / 2)
zmargin = zscore * squared_errors.std(ddof=1) / np.sqrt(m)
np.sqrt(mean - zmargin), np.sqrt(mean + zmargin)
```




    (45685.717918136594, 49690.68623889426)



# Extra material

## A full pipeline with both preparation and prediction


```python
full_pipeline_with_predictor = Pipeline([
        ("preparation", full_pipeline),
        ("linear", LinearRegression())
    ])

full_pipeline_with_predictor.fit(housing, housing_labels)
full_pipeline_with_predictor.predict(some_data)
```




    array([210644.60459286, 317768.80697211, 210956.43331178,  59218.98886849,
           189747.55849879])



## Model persistence using joblib


```python
my_model = full_pipeline_with_predictor
```


```python
import joblib
joblib.dump(my_model, "my_model.pkl") # DIFF
#...
my_model_loaded = joblib.load("my_model.pkl") # DIFF
```

## Example SciPy distributions for `RandomizedSearchCV`


```python
from scipy.stats import geom, expon
geom_distrib=geom(0.5).rvs(10000, random_state=42)
expon_distrib=expon(scale=1).rvs(10000, random_state=42)
plt.hist(geom_distrib, bins=50)
plt.show()
plt.hist(expon_distrib, bins=50)
plt.show()
```


    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_152_0.png)
    



    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_152_1.png)
    


# Exercise solutions

## 1.

Question: Try a Support Vector Machine regressor (`sklearn.svm.SVR`), with various hyperparameters such as `kernel="linear"` (with various values for the `C` hyperparameter) or `kernel="rbf"` (with various values for the `C` and `gamma` hyperparameters). Don't worry about what these hyperparameters mean for now. How does the best `SVR` predictor perform?

**Warning**: the following cell may take close to 30 minutes to run, or more depending on your hardware.


```python
from sklearn.model_selection import GridSearchCV

param_grid = [
        {'kernel': ['linear'], 'C': [10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
        {'kernel': ['rbf'], 'C': [1.0, 3.0, 10., 30., 100., 300., 1000.0],
         'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
    ]

svm_reg = SVR()
grid_search = GridSearchCV(svm_reg, param_grid, cv=5, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(housing_prepared, housing_labels)
```

    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    [CV] C=10.0, kernel=linear ...........................................


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV] ............................ C=10.0, kernel=linear, total=   3.9s
    [CV] C=10.0, kernel=linear ...........................................


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    3.9s remaining:    0.0s


    [CV] ............................ C=10.0, kernel=linear, total=   3.9s
    [CV] C=10.0, kernel=linear ...........................................
    [CV] ............................ C=10.0, kernel=linear, total=   4.6s
    [CV] C=10.0, kernel=linear ...........................................
    [CV] ............................ C=10.0, kernel=linear, total=   4.2s
    [CV] C=10.0, kernel=linear ...........................................
    [CV] ............................ C=10.0, kernel=linear, total=   4.5s
    [CV] C=30.0, kernel=linear ...........................................
    [CV] ............................ C=30.0, kernel=linear, total=   4.1s
    [CV] C=30.0, kernel=linear ...........................................
    [CV] ............................ C=30.0, kernel=linear, total=   4.2s
    [CV] C=30.0, kernel=linear ...........................................
    [CV] ............................ C=30.0, kernel=linear, total=   4.3s
    [CV] C=30.0, kernel=linear ...........................................
    [CV] ............................ C=30.0, kernel=linear, total=   4.0s
    [CV] C=30.0, kernel=linear ...........................................
    [CV] ............................ C=30.0, kernel=linear, total=   3.9s
    [CV] C=100.0, kernel=linear ..........................................
    [CV] ........................... C=100.0, kernel=linear, total=   3.9s
    [CV] C=100.0, kernel=linear ..........................................
    [CV] ........................... C=100.0, kernel=linear, total=   4.0s
    [CV] C=100.0, kernel=linear ..........................................
    [CV] ........................... C=100.0, kernel=linear, total=   4.0s
    [CV] C=100.0, kernel=linear ..........................................
    [CV] ........................... C=100.0, kernel=linear, total=   4.0s
    [CV] C=100.0, kernel=linear ..........................................
    [CV] ........................... C=100.0, kernel=linear, total=   3.9s
    [CV] C=300.0, kernel=linear ..........................................
    [CV] ........................... C=300.0, kernel=linear, total=   4.1s
    <<434 more lines>>
    [CV] C=1000.0, gamma=0.1, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.1, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=0.1, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.1, kernel=rbf, total=   6.8s
    [CV] C=1000.0, gamma=0.3, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.3, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=0.3, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.3, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=0.3, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.3, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=0.3, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.3, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=0.3, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=0.3, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=1.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=1.0, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=1.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=1.0, kernel=rbf, total=   6.8s
    [CV] C=1000.0, gamma=1.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=1.0, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=1.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=1.0, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=1.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=1.0, kernel=rbf, total=   6.7s
    [CV] C=1000.0, gamma=3.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=3.0, kernel=rbf, total=   7.4s
    [CV] C=1000.0, gamma=3.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=3.0, kernel=rbf, total=   7.4s
    [CV] C=1000.0, gamma=3.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=3.0, kernel=rbf, total=   7.4s
    [CV] C=1000.0, gamma=3.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=3.0, kernel=rbf, total=   7.4s
    [CV] C=1000.0, gamma=3.0, kernel=rbf .................................
    [CV] .................. C=1000.0, gamma=3.0, kernel=rbf, total=   7.3s


    [Parallel(n_jobs=1)]: Done 250 out of 250 | elapsed: 26.4min finished





    GridSearchCV(cv=5, estimator=SVR(),
                 param_grid=[{'C': [10.0, 30.0, 100.0, 300.0, 1000.0, 3000.0,
                                    10000.0, 30000.0],
                              'kernel': ['linear']},
                             {'C': [1.0, 3.0, 10.0, 30.0, 100.0, 300.0, 1000.0],
                              'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0],
                              'kernel': ['rbf']}],
                 scoring='neg_mean_squared_error', verbose=2)



The best model achieves the following score (evaluated using 5-fold cross validation):


```python
negative_mse = grid_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
```




    70363.84006944533



That's much worse than the `RandomForestRegressor`. Let's check the best hyperparameters found:


```python
grid_search.best_params_
```




    {'C': 30000.0, 'kernel': 'linear'}



The linear kernel seems better than the RBF kernel. Notice that the value of `C` is the maximum tested value. When this happens you definitely want to launch the grid search again with higher values for `C` (removing the smallest values), because it is likely that higher values of `C` will be better.

## 2.

Question: Try replacing `GridSearchCV` with `RandomizedSearchCV`.

**Warning**: the following cell may take close to 45 minutes to run, or more depending on your hardware.


```python
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import expon, reciprocal

# see https://docs.scipy.org/doc/scipy/reference/stats.html
# for `expon()` and `reciprocal()` documentation and more probability distribution functions.

# Note: gamma is ignored when kernel is "linear"
param_distribs = {
        'kernel': ['linear', 'rbf'],
        'C': reciprocal(20, 200000),
        'gamma': expon(scale=1.0),
    }

svm_reg = SVR()
rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
                                n_iter=50, cv=5, scoring='neg_mean_squared_error',
                                verbose=2, random_state=42)
rnd_search.fit(housing_prepared, housing_labels)
```

    Fitting 5 folds for each of 50 candidates, totalling 250 fits
    [CV] C=629.782329591372, gamma=3.010121430917521, kernel=linear ......


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  C=629.782329591372, gamma=3.010121430917521, kernel=linear, total=   4.2s
    [CV] C=629.782329591372, gamma=3.010121430917521, kernel=linear ......


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    4.2s remaining:    0.0s


    [CV]  C=629.782329591372, gamma=3.010121430917521, kernel=linear, total=   4.0s
    [CV] C=629.782329591372, gamma=3.010121430917521, kernel=linear ......
    [CV]  C=629.782329591372, gamma=3.010121430917521, kernel=linear, total=   4.5s
    [CV] C=629.782329591372, gamma=3.010121430917521, kernel=linear ......
    [CV]  C=629.782329591372, gamma=3.010121430917521, kernel=linear, total=   4.5s
    [CV] C=629.782329591372, gamma=3.010121430917521, kernel=linear ......
    [CV]  C=629.782329591372, gamma=3.010121430917521, kernel=linear, total=   4.3s
    [CV] C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf ......
    [CV]  C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf, total=   8.6s
    [CV] C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf ......
    [CV]  C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf, total=   9.1s
    [CV] C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf ......
    [CV]  C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf, total=   8.8s
    [CV] C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf ......
    [CV]  C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf, total=   8.9s
    [CV] C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf ......
    [CV]  C=26290.206464300216, gamma=0.9084469696321253, kernel=rbf, total=   9.0s
    [CV] C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf .....
    [CV]  C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf, total=   7.0s
    [CV] C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf .....
    [CV]  C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf, total=   7.0s
    [CV] C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf .....
    [CV]  C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf, total=   6.9s
    [CV] C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf .....
    [CV]  C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf, total=   7.0s
    [CV] C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf .....
    [CV]  C=84.14107900575871, gamma=0.059838768608680676, kernel=rbf, total=   7.0s
    [CV] C=432.37884813148855, gamma=0.15416196746656105, kernel=linear ..
    [CV]  C=432.37884813148855, gamma=0.15416196746656105, kernel=linear, total=   4.6s
    <<434 more lines>>
    [CV] C=61217.04421344494, gamma=1.6279689407405564, kernel=rbf .......
    [CV]  C=61217.04421344494, gamma=1.6279689407405564, kernel=rbf, total=  25.2s
    [CV] C=61217.04421344494, gamma=1.6279689407405564, kernel=rbf .......
    [CV]  C=61217.04421344494, gamma=1.6279689407405564, kernel=rbf, total=  23.2s
    [CV] C=926.9787684096649, gamma=2.147979593060577, kernel=rbf ........
    [CV]  C=926.9787684096649, gamma=2.147979593060577, kernel=rbf, total=   5.7s
    [CV] C=926.9787684096649, gamma=2.147979593060577, kernel=rbf ........
    [CV]  C=926.9787684096649, gamma=2.147979593060577, kernel=rbf, total=   5.7s
    [CV] C=926.9787684096649, gamma=2.147979593060577, kernel=rbf ........
    [CV]  C=926.9787684096649, gamma=2.147979593060577, kernel=rbf, total=   5.7s
    [CV] C=926.9787684096649, gamma=2.147979593060577, kernel=rbf ........
    [CV]  C=926.9787684096649, gamma=2.147979593060577, kernel=rbf, total=   5.8s
    [CV] C=926.9787684096649, gamma=2.147979593060577, kernel=rbf ........
    [CV]  C=926.9787684096649, gamma=2.147979593060577, kernel=rbf, total=   5.6s
    [CV] C=33946.157064934, gamma=2.2642426492862313, kernel=linear ......
    [CV]  C=33946.157064934, gamma=2.2642426492862313, kernel=linear, total=  10.0s
    [CV] C=33946.157064934, gamma=2.2642426492862313, kernel=linear ......
    [CV]  C=33946.157064934, gamma=2.2642426492862313, kernel=linear, total=   9.7s
    [CV] C=33946.157064934, gamma=2.2642426492862313, kernel=linear ......
    [CV]  C=33946.157064934, gamma=2.2642426492862313, kernel=linear, total=   8.9s
    [CV] C=33946.157064934, gamma=2.2642426492862313, kernel=linear ......
    [CV]  C=33946.157064934, gamma=2.2642426492862313, kernel=linear, total=  10.4s
    [CV] C=33946.157064934, gamma=2.2642426492862313, kernel=linear ......
    [CV]  C=33946.157064934, gamma=2.2642426492862313, kernel=linear, total=   9.3s
    [CV] C=84789.82947739525, gamma=0.3176359085304841, kernel=linear ....
    [CV]  C=84789.82947739525, gamma=0.3176359085304841, kernel=linear, total=  25.8s
    [CV] C=84789.82947739525, gamma=0.3176359085304841, kernel=linear ....
    [CV]  C=84789.82947739525, gamma=0.3176359085304841, kernel=linear, total=  18.5s
    [CV] C=84789.82947739525, gamma=0.3176359085304841, kernel=linear ....
    [CV]  C=84789.82947739525, gamma=0.3176359085304841, kernel=linear, total=  28.3s
    [CV] C=84789.82947739525, gamma=0.3176359085304841, kernel=linear ....
    [CV]  C=84789.82947739525, gamma=0.3176359085304841, kernel=linear, total=  20.8s
    [CV] C=84789.82947739525, gamma=0.3176359085304841, kernel=linear ....
    [CV]  C=84789.82947739525, gamma=0.3176359085304841, kernel=linear, total=  15.6s


    [Parallel(n_jobs=1)]: Done 250 out of 250 | elapsed: 44.0min finished





    RandomizedSearchCV(cv=5, estimator=SVR(), n_iter=50,
                       param_distributions={'C': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f9bd002c790>,
                                            'gamma': <scipy.stats._distn_infrastructure.rv_frozen object at 0x7f9bd002cb10>,
                                            'kernel': ['linear', 'rbf']},
                       random_state=42, scoring='neg_mean_squared_error',
                       verbose=2)



The best model achieves the following score (evaluated using 5-fold cross validation):


```python
negative_mse = rnd_search.best_score_
rmse = np.sqrt(-negative_mse)
rmse
```




    54767.960710084146



Now this is much closer to the performance of the `RandomForestRegressor` (but not quite there yet). Let's check the best hyperparameters found:


```python
rnd_search.best_params_
```




    {'C': 157055.10989448498, 'gamma': 0.26497040005002437, 'kernel': 'rbf'}



This time the search found a good set of hyperparameters for the RBF kernel. Randomized search tends to find better hyperparameters than grid search in the same amount of time.

Let's look at the exponential distribution we used, with `scale=1.0`. Note that some samples are much larger or smaller than 1.0, but when you look at the log of the distribution, you can see that most values are actually concentrated roughly in the range of exp(-2) to exp(+2), which is about 0.1 to 7.4.


```python
expon_distrib = expon(scale=1.)
samples = expon_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Exponential distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
```


    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_173_0.png)
    


The distribution we used for `C` looks quite different: the scale of the samples is picked from a uniform distribution within a given range, which is why the right graph, which represents the log of the samples, looks roughly constant. This distribution is useful when you don't have a clue of what the target scale is:


```python
reciprocal_distrib = reciprocal(20, 200000)
samples = reciprocal_distrib.rvs(10000, random_state=42)
plt.figure(figsize=(10, 4))
plt.subplot(121)
plt.title("Reciprocal distribution (scale=1.0)")
plt.hist(samples, bins=50)
plt.subplot(122)
plt.title("Log of this distribution")
plt.hist(np.log(samples), bins=50)
plt.show()
```


    
![png](02_end_to_end_machine_learning_project_files/02_end_to_end_machine_learning_project_175_0.png)
    


The reciprocal distribution is useful when you have no idea what the scale of the hyperparameter should be (indeed, as you can see on the figure on the right, all scales are equally likely, within the given range), whereas the exponential distribution is best when you know (more or less) what the scale of the hyperparameter should be.

## 3.

Question: Try adding a transformer in the preparation pipeline to select only the most important attributes.


```python
from sklearn.base import BaseEstimator, TransformerMixin

def indices_of_top_k(arr, k):
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])

class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importances = feature_importances
        self.k = k
    def fit(self, X, y=None):
        self.feature_indices_ = indices_of_top_k(self.feature_importances, self.k)
        return self
    def transform(self, X):
        return X[:, self.feature_indices_]
```

Note: this feature selector assumes that you have already computed the feature importances somehow (for example using a `RandomForestRegressor`). You may be tempted to compute them directly in the `TopFeatureSelector`'s `fit()` method, however this would likely slow down grid/randomized search since the feature importances would have to be computed for every hyperparameter combination (unless you implement some sort of cache).

Let's define the number of top features we want to keep:


```python
k = 5
```

Now let's look for the indices of the top k features:


```python
top_k_feature_indices = indices_of_top_k(feature_importances, k)
top_k_feature_indices
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/2341415780.py in <module>
    ----> 1 top_k_feature_indices = indices_of_top_k(feature_importances, k)
          2 top_k_feature_indices


    NameError: name 'feature_importances' is not defined



```python
np.array(attributes)[top_k_feature_indices]
```


    ---------------------------------------------------------------------------

    NameError                                 Traceback (most recent call last)

    /var/folders/5h/8nlly35s0c77l8djfpgthddc0000gn/T/ipykernel_30582/2431309223.py in <module>
    ----> 1 np.array(attributes)[top_k_feature_indices]
    

    NameError: name 'top_k_feature_indices' is not defined


Let's double check that these are indeed the top k features:


```python
sorted(zip(feature_importances, attributes), reverse=True)[:k]
```




    [(0.36615898061813423, 'median_income'),
     (0.16478099356159054, 'INLAND'),
     (0.10879295677551575, 'pop_per_hhold'),
     (0.07334423551601243, 'longitude'),
     (0.06290907048262032, 'latitude')]



Looking good... Now let's create a new pipeline that runs the previously defined preparation pipeline, and adds top k feature selection:


```python
preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])
```


```python
housing_prepared_top_k_features = preparation_and_feature_selection_pipeline.fit_transform(housing)
```

Let's look at the features of the first 3 instances:


```python
housing_prepared_top_k_features[0:3]
```




    array([[-1.15604281,  0.77194962, -0.61493744, -0.08649871,  0.        ],
           [-1.17602483,  0.6596948 ,  1.33645936, -0.03353391,  0.        ],
           [ 1.18684903, -1.34218285, -0.5320456 , -0.09240499,  0.        ]])



Now let's double check that these are indeed the top k features:


```python
housing_prepared[0:3, top_k_feature_indices]
```




    array([[-1.15604281,  0.77194962, -0.61493744, -0.08649871,  0.        ],
           [-1.17602483,  0.6596948 ,  1.33645936, -0.03353391,  0.        ],
           [ 1.18684903, -1.34218285, -0.5320456 , -0.09240499,  0.        ]])



Works great!  :)

## 4.

Question: Try creating a single pipeline that does the full data preparation plus the final prediction.


```python
prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('svm_reg', SVR(**rnd_search.best_params_))
])
```


```python
prepare_select_and_predict_pipeline.fit(housing, housing_labels)
```




    Pipeline(steps=[('preparation',
                     ColumnTransformer(transformers=[('num',
                                                      Pipeline(steps=[('imputer',
                                                                       SimpleImputer(strategy='median')),
                                                                      ('attribs_adder',
                                                                       CombinedAttributesAdder()),
                                                                      ('std_scaler',
                                                                       StandardScaler())]),
                                                      ['longitude', 'latitude',
                                                       'housing_median_age',
                                                       'total_rooms',
                                                       'total_bedrooms',
                                                       'population', 'households',
                                                       'median_income']),
                                                     ('cat', OneHotEncoder(...
                     TopFeatureSelector(feature_importances=array([7.33442355e-02, 6.29090705e-02, 4.11437985e-02, 1.46726854e-02,
           1.41064835e-02, 1.48742809e-02, 1.42575993e-02, 3.66158981e-01,
           5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
           1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03]),
                                        k=5)),
                    ('svm_reg',
                     SVR(C=157055.10989448498, gamma=0.26497040005002437))])



Let's try the full pipeline on a few instances:


```python
some_data = housing.iloc[:4]
some_labels = housing_labels.iloc[:4]

print("Predictions:\t", prepare_select_and_predict_pipeline.predict(some_data))
print("Labels:\t\t", list(some_labels))
```

    Predictions:	 [203214.28978849 371846.88152572 173295.65441612  47328.3970888 ]
    Labels:		 [286600.0, 340600.0, 196900.0, 46300.0]


Well, the full pipeline seems to work fine. Of course, the predictions are not fantastic: they would be better if we used the best `RandomForestRegressor` that we found earlier, rather than the best `SVR`.

## 5.

Question: Automatically explore some preparation options using `GridSearchCV`.

**Warning**: the following cell may take close to 45 minutes to run, or more depending on your hardware.


```python
param_grid = [{
    'preparation__num__imputer__strategy': ['mean', 'median', 'most_frequent'],
    'feature_selection__k': list(range(1, len(feature_importances) + 1))
}]

grid_search_prep = GridSearchCV(prepare_select_and_predict_pipeline, param_grid, cv=5,
                                scoring='neg_mean_squared_error', verbose=2)
grid_search_prep.fit(housing, housing_labels)
```

    Fitting 5 folds for each of 48 candidates, totalling 240 fits
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=mean 


    [Parallel(n_jobs=1)]: Using backend SequentialBackend with 1 concurrent workers.


    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=mean, total=   4.2s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=mean 


    [Parallel(n_jobs=1)]: Done   1 out of   1 | elapsed:    4.2s remaining:    0.0s


    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=mean, total=   5.2s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=mean, total=   4.7s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=mean, total=   4.7s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=mean, total=   4.8s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=median, total=   5.1s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=median, total=   4.9s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=median, total=   4.7s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=median, total=   4.3s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=median, total=   4.2s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=most_frequent, total=   4.6s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=most_frequent, total=   4.3s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=most_frequent, total=   4.4s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=most_frequent, total=   4.7s
    [CV] feature_selection__k=1, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=1, preparation__num__imputer__strategy=most_frequent, total=   4.8s
    [CV] feature_selection__k=2, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=2, preparation__num__imputer__strategy=mean, total=   4.8s
    <<414 more lines>>
    [CV] feature_selection__k=15, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=15, preparation__num__imputer__strategy=most_frequent, total=  15.8s
    [CV] feature_selection__k=15, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=15, preparation__num__imputer__strategy=most_frequent, total=  19.8s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=mean, total=  17.9s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=mean, total=  19.2s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=mean, total=  18.2s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=mean, total=  19.1s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=mean 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=mean, total=  16.4s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=median, total=  17.9s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=median, total=  19.2s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=median, total=  20.5s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=median, total=  17.1s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=median 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=median, total=  20.3s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=most_frequent, total=  16.7s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=most_frequent, total=  19.4s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=most_frequent, total=  17.2s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=most_frequent, total=  17.5s
    [CV] feature_selection__k=16, preparation__num__imputer__strategy=most_frequent 
    [CV]  feature_selection__k=16, preparation__num__imputer__strategy=most_frequent, total=  19.1s


    [Parallel(n_jobs=1)]: Done 240 out of 240 | elapsed: 42.3min finished





    GridSearchCV(cv=5,
                 estimator=Pipeline(steps=[('preparation',
                                            ColumnTransformer(transformers=[('num',
                                                                             Pipeline(steps=[('imputer',
                                                                                              SimpleImputer(strategy='median')),
                                                                                             ('attribs_adder',
                                                                                              CombinedAttributesAdder()),
                                                                                             ('std_scaler',
                                                                                              StandardScaler())]),
                                                                             ['longitude',
                                                                              'latitude',
                                                                              'housing_median_age',
                                                                              'total_rooms',
                                                                              'total_bedrooms',
                                                                              'population',
                                                                              'households',
                                                                              'median_inc...
           5.64191792e-02, 1.08792957e-01, 5.33510773e-02, 1.03114883e-02,
           1.64780994e-01, 6.02803867e-05, 1.96041560e-03, 2.85647464e-03]),
                                                               k=5)),
                                           ('svm_reg',
                                            SVR(C=157055.10989448498,
                                                gamma=0.26497040005002437))]),
                 param_grid=[{'feature_selection__k': [1, 2, 3, 4, 5, 6, 7, 8, 9,
                                                       10, 11, 12, 13, 14, 15, 16],
                              'preparation__num__imputer__strategy': ['mean',
                                                                      'median',
                                                                      'most_frequent']}],
                 scoring='neg_mean_squared_error', verbose=2)




```python
grid_search_prep.best_params_
```




    {'feature_selection__k': 15,
     'preparation__num__imputer__strategy': 'most_frequent'}



The best imputer strategy is `most_frequent` and apparently almost all features are useful (15 out of 16). The last one (`ISLAND`) seems to just add some noise.

Congratulations! You already know quite a lot about Machine Learning. :)
