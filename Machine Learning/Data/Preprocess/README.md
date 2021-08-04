# Data Precessing

**Once we know our data briefly, we can preprocess our data, which may fast our training, improve model performance and so on**
**Assuming that our data is stored in pandas Dataframe format, and the following code is based on it.**


&nbsp;
#### .cut()
&emsp;
We can use the ``pd.cut()`` function to create a new category, which is dividedan median_income category into 5 groups (labeled from 1 to 5): 
category 1 ranges from 0 to 1.5 (i.e., less than $15,000), 
category 2 from 1.5 to 3, and so on


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

![image](https://github.com/dynotw/Basic-Machine-Learning-Code/blob/main/Machine%20Learning/Data/Preview/02_end_to_end_machine_learning_project_31_1.png)
