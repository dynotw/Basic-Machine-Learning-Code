
```python
def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
```
## Visualization

After we have our data or result, sometimes we need to visualize them to have a direct view.


### Image Data

#### Single Image
```python
def plot_digit(data):
    image = data.reshape(28, 28)
    # Here're grey image, so cmap = binary. 
    # For RGB data, we can ignore this parameter
    plt.imshow(image, cmap = mpl.cm.binary,
               interpolation="nearest")
    plt.axis("off")
```


#### Group Images
```python
# EXTRA
def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size,size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row : (row + 1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    
    # Here're grey image, so cmap = binary. 
    # For RGB data, we can ignore this parameter
    plt.imshow(image, cmap = mpl.cm.binary, **options)
    plt.axis("off")
```


```python
plt.figure(figsize=(9,9))
example_images = X[:100]
plot_digits(example_images, images_per_row=10)
save_fig("more_digits_plot")
plt.show()
```

    
![image](https://github.com/dynotw/Basic-Machine-Learning-Code/blob/main/visualization/03_classification_files/03_classification_16_1.png)
