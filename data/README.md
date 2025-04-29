## Data

The data used to train the model for this project is, Large Movie Review Dataset, by Stanford University. 

This is a dataset for binary sentiment classification containing substantially more data than previous benchmark datasets. Providing a set of 25,000 highly polar movie reviews for training, and 25,000 for testing. There is additional unlabeled data for use as well. Raw text and already processed bag of words formats are provided.

### Download

This dataset can be directly download from [Large Movie Review Dataset](https://ai.stanford.edu/~amaas/data/sentiment/).

You can also use the following block of code used in our preprocessing step to download the dataset:

```python
def download_imdb():
    url = "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz"
    filename = "aclImdb_v1.tar.gz"
    foldername = "../data/aclImdb"

    if not os.path.exists(foldername):
        print("Downloading IMDb dataset...")
        urllib.request.urlretrieve(url, filename)
        print("Extracting...")
        with tarfile.open(filename, "r:gz") as tar:
            tar.extractall()
        print("Done!")
    else:
        print("IMDb dataset already exists.")
```