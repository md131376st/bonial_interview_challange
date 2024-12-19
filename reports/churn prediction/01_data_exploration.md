# Goal
1. load row data
2. convert text data to correct datatype
3. show static's about data
3. visualize row data ( understand the data we are using)


```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

```


```python
BASE_DIR = os.path.abspath("..")
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset")

```


```python
installs = pd.read_csv(os.path.join(RAW_DATA_PATH, "installs.txt"), sep="\t")
brochure_views = pd.read_csv(os.path.join(RAW_DATA_PATH, "brochure views.txt"), sep="\t")
brochure_views_july = pd.read_csv(os.path.join(RAW_DATA_PATH, "brochure views july.txt"), sep="\t")
app_starts = pd.read_csv(os.path.join(RAW_DATA_PATH, "app starts.txt"), sep="\t")
app_starts_july = pd.read_csv(os.path.join(RAW_DATA_PATH, "app starts july.txt"), sep="\t")
```

## convert datatypes


```python
print(installs.info())
print(brochure_views.info())
print(app_starts.info())
```

    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 20000 entries, 0 to 19999
    Data columns (total 6 columns):
     #   Column       Non-Null Count  Dtype 
    ---  ------       --------------  ----- 
     0   id           20000 non-null  int64 
     1   InstallDate  20000 non-null  object
     2   productId    20000 non-null  object
     3   userId       20000 non-null  object
     4   model        19963 non-null  object
     5   campaignId   20000 non-null  object
    dtypes: int64(1), object(5)
    memory usage: 937.6+ KB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 279213 entries, 0 to 279212
    Data columns (total 6 columns):
     #   Column           Non-Null Count   Dtype  
    ---  ------           --------------   -----  
     0   id               279213 non-null  int64  
     1   userId           279213 non-null  object 
     2   dateCreated      279213 non-null  object 
     3   page_turn_count  279213 non-null  int64  
     4   view_duration    269602 non-null  float64
     5   brochure_id      279213 non-null  int64  
    dtypes: float64(1), int64(3), object(2)
    memory usage: 12.8+ MB
    None
    <class 'pandas.core.frame.DataFrame'>
    RangeIndex: 144104 entries, 0 to 144103
    Data columns (total 2 columns):
     #   Column       Non-Null Count   Dtype 
    ---  ------       --------------   ----- 
     0   dateCreated  144104 non-null  object
     1   userId       144104 non-null  object
    dtypes: object(2)
    memory usage: 2.2+ MB
    None
    


```python
installs['InstallDate'] = pd.to_datetime(installs['InstallDate'], errors='coerce')
brochure_views['dateCreated'] = pd.to_datetime(brochure_views['dateCreated'], errors='coerce')
brochure_views_july['dateCreated'] = pd.to_datetime(brochure_views_july['dateCreated'], errors='coerce')
app_starts['dateCreated'] = pd.to_datetime(app_starts['dateCreated'], errors='coerce')
app_starts_july['dateCreated'] = pd.to_datetime(app_starts_july['dateCreated'], errors='coerce')
```

## data overview




```python
print("Data Shapes:")
print("Installs:", installs.shape)
print("Brochure Views:", brochure_views.shape)
print("Brochure Views July (Test):", brochure_views_july.shape)
print("App Starts:", app_starts.shape)
print("App Starts July (Test):", app_starts_july.shape)


```

    Data Shapes:
    Installs: (20000, 6)
    Brochure Views: (279213, 6)
    Brochure Views July (Test): (74431, 6)
    App Starts: (144104, 2)
    App Starts July (Test): (39969, 2)
    


```python
print("\nPreview of Installs:")
display(installs.head())

print("\nPreview of Brochure Views:")
display(brochure_views.head())

print("\nPreview of App Starts:")
display(app_starts.head())

```

    
    Preview of Installs:
    


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
      <th>id</th>
      <th>InstallDate</th>
      <th>productId</th>
      <th>userId</th>
      <th>model</th>
      <th>campaignId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>37371890</td>
      <td>2017-05-04 10:06:27.807</td>
      <td>de.kaufda.kaufda</td>
      <td>5fc13850-de51-4426-96ce-72aaec895abb</td>
      <td>ipad2,7</td>
      <td>000000</td>
    </tr>
    <tr>
      <th>1</th>
      <td>36979918</td>
      <td>2017-04-24 09:27:16.173</td>
      <td>com-bonial-kaufda</td>
      <td>b13a035e-e9bf-49db-8f93-b49d491bef53</td>
      <td>sm-g800f</td>
      <td>3iikhy</td>
    </tr>
    <tr>
      <th>2</th>
      <td>37371070</td>
      <td>2017-05-04 16:05:09.568</td>
      <td>de.kaufda.kaufda</td>
      <td>5deb0aad-43bb-4ee1-868b-98f36c3d5bbf</td>
      <td>iphone8,4</td>
      <td>tl3cy8</td>
    </tr>
    <tr>
      <th>3</th>
      <td>36897929</td>
      <td>2017-04-22 13:18:18.437</td>
      <td>com-bonial-kaufda</td>
      <td>64707e97-d5e1-4622-a89a-bebf8432fd33</td>
      <td>sm-g955f</td>
      <td>000000</td>
    </tr>
    <tr>
      <th>4</th>
      <td>36606053</td>
      <td>2017-04-15 14:03:51.199</td>
      <td>com-bonial-kaufda</td>
      <td>770a391d-eda5-423b-b672-845f1e12661f</td>
      <td>sm-j320h</td>
      <td>000000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Preview of Brochure Views:
    


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
      <th>id</th>
      <th>userId</th>
      <th>dateCreated</th>
      <th>page_turn_count</th>
      <th>view_duration</th>
      <th>brochure_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>6269194661</td>
      <td>9491a960-206a-4a58-9177-e78cb1f05e70</td>
      <td>2017-04-30 23:47:09.539</td>
      <td>30</td>
      <td>95172.0</td>
      <td>672239440</td>
    </tr>
    <tr>
      <th>1</th>
      <td>6269192581</td>
      <td>9491a960-206a-4a58-9177-e78cb1f05e70</td>
      <td>2017-04-30 23:47:00.697</td>
      <td>2</td>
      <td>4000.0</td>
      <td>673861625</td>
    </tr>
    <tr>
      <th>2</th>
      <td>6269188351</td>
      <td>9491a960-206a-4a58-9177-e78cb1f05e70</td>
      <td>2017-04-30 23:46:39.917</td>
      <td>1</td>
      <td>14000.0</td>
      <td>660764240</td>
    </tr>
    <tr>
      <th>3</th>
      <td>6204426632</td>
      <td>fa623647-dfc9-49b4-bbb6-77cbacd599f4</td>
      <td>2017-04-30 23:46:13.597</td>
      <td>64</td>
      <td>168000.0</td>
      <td>672658544</td>
    </tr>
    <tr>
      <th>4</th>
      <td>6204424112</td>
      <td>9491a960-206a-4a58-9177-e78cb1f05e70</td>
      <td>2017-04-30 23:46:02.197</td>
      <td>1</td>
      <td>33000.0</td>
      <td>660764240</td>
    </tr>
  </tbody>
</table>
</div>


    
    Preview of App Starts:
    


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
      <th>dateCreated</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-06-30 14:14:54.793</td>
      <td>50e72534-a4f4-40d7-96d5-ecbe4eb314e9</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-06-30 14:03:13.010</td>
      <td>b3712849-595e-403f-84d2-4698439056b0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-06-28 16:26:48.383</td>
      <td>99cea50b-3ecf-4102-8290-997eaf32a6b6</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-06-27 10:23:29.943</td>
      <td>78c06433-9ea8-4835-aff5-f64b262d0fb4</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-06-27 10:11:27.634</td>
      <td>510c5f9e-de54-45ee-909c-c14103130e5e</td>
    </tr>
  </tbody>
</table>
</div>



```python
print("Missing Values:")
print("Missing in Installs:\n", installs.isna().sum())
print("Missing in Brochure Views:\n", brochure_views.isna().sum())
print("Missing in App Starts:\n", app_starts.isna().sum())
```

    Missing Values:
    Missing in Installs:
     id              0
    InstallDate     0
    productId       0
    userId          0
    model          37
    campaignId      0
    dtype: int64
    Missing in Brochure Views:
     id                    0
    userId                0
    dateCreated           0
    page_turn_count       0
    view_duration      9611
    brochure_id           0
    dtype: int64
    Missing in App Starts:
     dateCreated    0
    userId         0
    dtype: int64
    

### duration analysis on row data
after viewing the data on brochure_views it was observed that the view_duration had missing values and data errors with negative duration



```python
column ='view_duration'
missing_duration = brochure_views[brochure_views[column].isna()]
missing_count = missing_duration.shape[0]
print("Number of rows with missing view_duration:", missing_count)
percentage_missing = (missing_count / brochure_views.shape[0]) * 100
print("Percentage of missing view_duration:", percentage_missing)
display(missing_duration.head())
print(f"min duration value: {brochure_views[column].min()}")
# finding positive min value for replacing as view duration
print(f"min positive duration value: {brochure_views[column][brochure_views[column]>0].min()}")

```

    Number of rows with missing view_duration: 9611
    Percentage of missing view_duration: 3.442174970363128
    


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
      <th>id</th>
      <th>userId</th>
      <th>dateCreated</th>
      <th>page_turn_count</th>
      <th>view_duration</th>
      <th>brochure_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>97</th>
      <td>6201454322</td>
      <td>9d4e761d-8858-4af2-ae72-c9d40f80d219</td>
      <td>2017-04-30 21:10:18.607</td>
      <td>1</td>
      <td>NaN</td>
      <td>671973171</td>
    </tr>
    <tr>
      <th>142</th>
      <td>6264968691</td>
      <td>6fe9afbc-784f-4d22-a952-d50e67261631</td>
      <td>2017-04-30 20:27:03.789</td>
      <td>1</td>
      <td>NaN</td>
      <td>671979651</td>
    </tr>
    <tr>
      <th>157</th>
      <td>6198746552</td>
      <td>0ad62dc6-de94-43cf-a2d5-6abae758bb85</td>
      <td>2017-04-30 19:02:54.391</td>
      <td>1</td>
      <td>NaN</td>
      <td>672239392</td>
    </tr>
    <tr>
      <th>162</th>
      <td>6197986142</td>
      <td>e691e2a3-db13-4d74-afe4-96e61f9af80c</td>
      <td>2017-04-30 17:59:49.046</td>
      <td>1</td>
      <td>NaN</td>
      <td>541614364</td>
    </tr>
    <tr>
      <th>212</th>
      <td>6200889782</td>
      <td>90af2ceb-aa34-417f-8a9b-69d6a1914f4b</td>
      <td>2017-04-30 20:51:47.284</td>
      <td>1</td>
      <td>NaN</td>
      <td>671802472</td>
    </tr>
  </tbody>
</table>
</div>


    min duration value: -18000.0
    min positive duration value: 1000.0
    


```python
negative_count = (brochure_views[column] < 0).sum()
percentage_negative_duration = (negative_count / brochure_views.shape[0]) * 100
print("Number of rows with negative view_duration:", negative_count)
print("Percentage of negative view_duration:", percentage_negative_duration)
```

    Number of rows with negative view_duration: 5
    Percentage of negative view_duration: 0.0017907475654786848
    


```python
plt.figure(figsize=(4,4))
sns.histplot(brochure_views['view_duration'], kde=True)
plt.title("Box Plot of Raw View Duration")
plt.tight_layout()
plt.show()
```


    
![png](reports/churn%20prediction/01_data_exploration_files/reports/churn%20prediction/01_data_exploration_14_0.png)
    



```python
# Checking page_turn_count distribution
plt.figure(figsize=(4,4))
sns.histplot(brochure_views['page_turn_count'], kde=True)
plt.title("Distribution of Page Turn Count (April-June)")
plt.xlabel("Page Turn Count")
plt.ylabel("Frequency")
plt.tight_layout()
plt.show()

```


    
![png](reports/churn%20prediction/01_data_exploration_files/reports/churn%20prediction/01_data_exploration_15_0.png)
    



```python
# view Duplicate values
print("Duplicates in installs:", installs.duplicated().sum())
print("Duplicates in brochure_views:", brochure_views.duplicated().sum())
print("Duplicates in app_starts:", app_starts.duplicated().sum())
```

    Duplicates in installs: 0
    Duplicates in brochure_views: 0
    Duplicates in app_starts: 52
    

### app_starts analysis


```python
# have a sample of duplicated values
duplicated_rows = app_starts[app_starts.duplicated(keep=False)]

print(f"Number of duplicated rows in app_starts: {duplicated_rows.shape[0]}")
display(duplicated_rows.head())
```

    Number of duplicated rows in app_starts: 103
    


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
      <th>dateCreated</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>47992</th>
      <td>2017-05-26 12:01:41.354</td>
      <td>df92b9d7-2a8b-41c6-85a7-1c1dd57d6bea</td>
    </tr>
    <tr>
      <th>47993</th>
      <td>2017-05-26 12:01:41.354</td>
      <td>df92b9d7-2a8b-41c6-85a7-1c1dd57d6bea</td>
    </tr>
    <tr>
      <th>71954</th>
      <td>2017-04-08 01:58:49.867</td>
      <td>8d9b31cd-879f-401d-8595-0ccfd52a01b4</td>
    </tr>
    <tr>
      <th>75595</th>
      <td>2017-04-06 01:24:08.140</td>
      <td>dbe3dc28-ce77-423e-8922-f9ff8494c702</td>
    </tr>
    <tr>
      <th>75596</th>
      <td>2017-04-06 01:24:08.140</td>
      <td>dbe3dc28-ce77-423e-8922-f9ff8494c702</td>
    </tr>
  </tbody>
</table>
</div>


### check for inconstancy app_start



```python

app_starts.drop_duplicates(inplace=True)
# get the earliest installation
user_earliest_install = installs.groupby('userId', as_index=False)['InstallDate'].min()
user_earliest_install.rename(columns={'InstallDate': 'earliest_install_date'}, inplace=True)

app_starts_install = app_starts.merge(user_earliest_install, on='userId', how='left')
app_starts_filtered = app_starts_install[app_starts_install['dateCreated'] >= app_starts_install['earliest_install_date']]
print("Original app_starts rows:", app_starts.shape[0])
print("Filtered app_starts rows:", app_starts_filtered.shape[0])
consistent_data = (app_starts_filtered.shape[0] / app_starts.shape[0]) * 100
print("Percentage of inconsistent app_starts:", consistent_data)
display(app_starts_filtered.head())
```

    Original app_starts rows: 144052
    Filtered app_starts rows: 139830
    Percentage of inconsistent app_starts: 97.06911393108044
    


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
      <th>dateCreated</th>
      <th>userId</th>
      <th>earliest_install_date</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-06-30 14:14:54.793</td>
      <td>50e72534-a4f4-40d7-96d5-ecbe4eb314e9</td>
      <td>2017-04-21 13:16:18.599</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-06-30 14:03:13.010</td>
      <td>b3712849-595e-403f-84d2-4698439056b0</td>
      <td>2017-04-28 15:28:20.546</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-06-28 16:26:48.383</td>
      <td>99cea50b-3ecf-4102-8290-997eaf32a6b6</td>
      <td>2017-05-09 20:45:19.723</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-06-27 10:23:29.943</td>
      <td>78c06433-9ea8-4835-aff5-f64b262d0fb4</td>
      <td>2017-05-15 13:29:37.377</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-06-27 10:11:27.634</td>
      <td>510c5f9e-de54-45ee-909c-c14103130e5e</td>
      <td>2017-05-07 08:03:04.573</td>
    </tr>
  </tbody>
</table>
</div>


### analyse install data


```python
installs_unique = installs.drop_duplicates(subset='userId', keep='first')
print("Number of  installs:", installs.shape[0])
print("Number of unique installs:", installs_unique.shape[0])
```

    Number of  installs: 20000
    Number of unique installs: 20000
    

### static summary



```python
print("\nSummary Statistics for Installs:")
display(installs.describe())

print("\nSummary Statistics for Brochure Views (April-June):")
display(brochure_views.describe())

print("\nSummary Statistics for App Starts (April-June):")
display(app_starts.describe())

# Summary statistics for categorical columns
print("\nCategorical Data Summary - Installs:")
display(installs.describe(include=['object']))

print("\nCategorical Data Summary - Brochure Views (April-June):")
display(brochure_views.describe(include=['object']))

print("\nCategorical Data Summary - App Starts (April-June):")
display(app_starts.describe(include=['object']))
```

    
    Summary Statistics for Installs:
    


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
      <th>id</th>
      <th>InstallDate</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.000000e+04</td>
      <td>20000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.726129e+07</td>
      <td>2017-05-01 10:11:51.664225024</td>
    </tr>
    <tr>
      <th>min</th>
      <td>3.598895e+07</td>
      <td>2017-04-01 00:07:03</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>3.665354e+07</td>
      <td>2017-04-16 10:49:46.854249984</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>3.724843e+07</td>
      <td>2017-05-01 14:05:37.252499968</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.790010e+07</td>
      <td>2017-05-16 07:25:37.047249920</td>
    </tr>
    <tr>
      <th>max</th>
      <td>3.850684e+07</td>
      <td>2017-05-31 23:45:33.995000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>7.106946e+05</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    Summary Statistics for Brochure Views (April-June):
    


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
      <th>id</th>
      <th>dateCreated</th>
      <th>page_turn_count</th>
      <th>view_duration</th>
      <th>brochure_id</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.792130e+05</td>
      <td>279213</td>
      <td>279213.000000</td>
      <td>2.696020e+05</td>
      <td>2.792130e+05</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.546832e+09</td>
      <td>2017-05-20 18:19:11.878093312</td>
      <td>15.830352</td>
      <td>8.737644e+04</td>
      <td>6.746679e+08</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.709034e+09</td>
      <td>2017-04-01 00:29:01.711000</td>
      <td>1.000000</td>
      <td>-1.800000e+04</td>
      <td>5.416142e+08</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.269143e+09</td>
      <td>2017-05-02 17:02:28.992999936</td>
      <td>1.000000</td>
      <td>1.100000e+04</td>
      <td>6.678938e+08</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.570289e+09</td>
      <td>2017-05-21 22:50:47.904999936</td>
      <td>9.000000</td>
      <td>3.800000e+04</td>
      <td>6.780668e+08</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.834928e+09</td>
      <td>2017-06-08 02:00:10.892999936</td>
      <td>23.000000</td>
      <td>1.014885e+05</td>
      <td>6.851689e+08</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.216487e+09</td>
      <td>2017-06-30 23:59:53.699000</td>
      <td>375.000000</td>
      <td>4.009200e+06</td>
      <td>6.995658e+08</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.642842e+08</td>
      <td>NaN</td>
      <td>18.688516</td>
      <td>1.512521e+05</td>
      <td>1.616983e+07</td>
    </tr>
  </tbody>
</table>
</div>


    
    Summary Statistics for App Starts (April-June):
    


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
      <th>dateCreated</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>144052</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>2017-05-21 04:31:01.851401984</td>
    </tr>
    <tr>
      <th>min</th>
      <td>2017-04-01 00:07:05</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>2017-05-03 12:04:01.976999936</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>2017-05-22 12:20:09.417499904</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>2017-06-08 16:06:03.944999936</td>
    </tr>
    <tr>
      <th>max</th>
      <td>2017-06-30 23:59:34.407000</td>
    </tr>
  </tbody>
</table>
</div>


    
    Categorical Data Summary - Installs:
    


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
      <th>productId</th>
      <th>userId</th>
      <th>model</th>
      <th>campaignId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>20000</td>
      <td>20000</td>
      <td>19963</td>
      <td>20000</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>3</td>
      <td>20000</td>
      <td>1210</td>
      <td>933</td>
    </tr>
    <tr>
      <th>top</th>
      <td>com-bonial-kaufda</td>
      <td>6135292b-9dd1-4dfe-8df9-59222c4a0f23</td>
      <td>gt-i9195</td>
      <td>000000</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>9726</td>
      <td>1</td>
      <td>1019</td>
      <td>13376</td>
    </tr>
  </tbody>
</table>
</div>


    
    Categorical Data Summary - Brochure Views (April-June):
    


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
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>279213</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>10943</td>
    </tr>
    <tr>
      <th>top</th>
      <td>9bf3ab9b-4f07-4e54-ac21-d92cf2afcea8</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>659</td>
    </tr>
  </tbody>
</table>
</div>


    
    Categorical Data Summary - App Starts (April-June):
    


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
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>144052</td>
    </tr>
    <tr>
      <th>unique</th>
      <td>20073</td>
    </tr>
    <tr>
      <th>top</th>
      <td>df22d1bd-2fb4-4353-82f1-50bed80a63cc</td>
    </tr>
    <tr>
      <th>freq</th>
      <td>438</td>
    </tr>
  </tbody>
</table>
</div>

