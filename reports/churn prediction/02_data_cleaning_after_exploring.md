# simple conclusions after first data explorations
1. brochure_views:
    - missing duration values
    - negative values
    - data duration, page_turn_count,  not normalized
2. app_starts:
    - duplicated values
    - inconsistent values with installs data

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath("../../notebooks/reports")
RAW_DATA_PATH = os.path.join(BASE_DIR, "dataset")
```


```python
installs = pd.read_csv(os.path.join(RAW_DATA_PATH, "installs.txt"), sep="\t")
brochure_views = pd.read_csv(os.path.join(RAW_DATA_PATH, "brochure views.txt"), sep="\t")
brochure_views_july = pd.read_csv(os.path.join(RAW_DATA_PATH, "brochure views july.txt"), sep="\t")
app_starts = pd.read_csv(os.path.join(RAW_DATA_PATH, "app starts.txt"), sep="\t")
app_starts_july = pd.read_csv(os.path.join(RAW_DATA_PATH, "app starts july.txt"), sep="\t")
```


```python

```


```python
installs['InstallDate'] = pd.to_datetime(installs['InstallDate'], errors='coerce')
brochure_views['dateCreated'] = pd.to_datetime(brochure_views['dateCreated'], errors='coerce')
brochure_views_july['dateCreated'] = pd.to_datetime(brochure_views_july['dateCreated'], errors='coerce')
app_starts['dateCreated'] = pd.to_datetime(app_starts['dateCreated'], errors='coerce')
app_starts_july['dateCreated'] = pd.to_datetime(app_starts_july['dateCreated'], errors='coerce')
```

# replacing missing and negative values with min nun negative value and normalization


```python

min_value = brochure_views['view_duration'][brochure_views['view_duration']>0].min()
brochure_views["view_duration"] = np.where(
    (brochure_views["view_duration"] < 0) | (brochure_views["view_duration"].isnull()),
    1000,
    brochure_views["view_duration"]
)
brochure_views_july["view_duration"] = np.where(
    (brochure_views_july["view_duration"] < 0) | (brochure_views_july["view_duration"].isnull()),
    1000,
    brochure_views_july["view_duration"]
)
normalize_col = ['view_duration', 'page_turn_count']
for col in normalize_col:
    brochure_views[f'{col}_log'] = np.log1p(brochure_views[col])
    brochure_views_july[f'{col}_log'] = np.log1p(brochure_views_july[col])
```

# remove duplications and inconsistent data


```python
app_starts.drop_duplicates(inplace=True)
app_starts = app_starts.merge(installs[['userId', 'InstallDate']], on='userId', how='left')
app_starts['dateCreated'] = app_starts[['dateCreated', 'InstallDate']].max(axis=1)
app_starts.drop(columns=['InstallDate'], inplace=True)
app_starts.head()
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
installs.head()
app_starts_july.head()

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
      <th>dateCreated</th>
      <th>userId</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2017-07-31 21:24:51.423</td>
      <td>c6adac02-336f-4cfb-9478-f7b822200215</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2017-07-31 21:10:24.981</td>
      <td>4b498ee2-2207-45d6-805e-929491a0bb6a</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2017-07-31 20:58:27.601</td>
      <td>e49dd902-816e-4e32-8320-9580340eaa1d</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2017-07-31 20:41:32.767</td>
      <td>b28aaf42-476f-489d-8b90-f430a3788c5a</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2017-07-29 16:51:55.217</td>
      <td>4bfb581c-a205-4189-bac1-c579fb92c575</td>
    </tr>
  </tbody>
</table>
</div>




```python
app_starts_july.drop_duplicates(inplace=True)
app_starts_july = app_starts_july.merge(installs[['userId', 'InstallDate']], on='userId', how='left')
app_starts_july['dateCreated'] = app_starts_july[['dateCreated', 'InstallDate']].max(axis=1)
app_starts_july.drop(columns=['InstallDate'], inplace=True)
```

# visualization after cleaning


```python
# impact of log transfer
brochure_views[['view_duration_log', 'page_turn_count_log']].hist(figsize=(10, 5))
```




    array([[<Axes: title={'center': 'view_duration_log'}>,
            <Axes: title={'center': 'page_turn_count_log'}>]], dtype=object)




    
![png](reports/churn%20prediction/02_data_cleaning_after_exploring_files/reports/churn%20prediction/02_data_cleaning_after_exploring_12_1.png)
    



```python
print(app_starts[app_starts['dateCreated'].isnull()].shape)
print(app_starts_july[app_starts_july['dateCreated'].isnull()].shape)
```

    (0, 2)
    (0, 2)
    


```python
print("\nBrochure Views Describe:\n")
display (brochure_views.describe())
```

    
    Brochure Views Describe:
    
    


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
      <th>view_duration_log</th>
      <th>page_turn_count_log</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>2.792130e+05</td>
      <td>279213</td>
      <td>279213.000000</td>
      <td>2.792130e+05</td>
      <td>2.792130e+05</td>
      <td>279213.000000</td>
      <td>279213.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>6.546832e+09</td>
      <td>2017-05-20 18:19:11.878093312</td>
      <td>15.830352</td>
      <td>8.440337e+04</td>
      <td>6.746679e+08</td>
      <td>10.282973</td>
      <td>2.187518</td>
    </tr>
    <tr>
      <th>min</th>
      <td>5.709034e+09</td>
      <td>2017-04-01 00:29:01.711000</td>
      <td>1.000000</td>
      <td>1.000000e+03</td>
      <td>5.416142e+08</td>
      <td>6.908755</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>6.269143e+09</td>
      <td>2017-05-02 17:02:28.992999936</td>
      <td>1.000000</td>
      <td>9.000000e+03</td>
      <td>6.678938e+08</td>
      <td>9.105091</td>
      <td>0.693147</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>6.570289e+09</td>
      <td>2017-05-21 22:50:47.904999936</td>
      <td>9.000000</td>
      <td>3.600000e+04</td>
      <td>6.780668e+08</td>
      <td>10.491302</td>
      <td>2.302585</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>6.834928e+09</td>
      <td>2017-06-08 02:00:10.892999936</td>
      <td>23.000000</td>
      <td>9.800000e+04</td>
      <td>6.851689e+08</td>
      <td>11.492733</td>
      <td>3.178054</td>
    </tr>
    <tr>
      <th>max</th>
      <td>7.216487e+09</td>
      <td>2017-06-30 23:59:53.699000</td>
      <td>375.000000</td>
      <td>4.009200e+06</td>
      <td>6.995658e+08</td>
      <td>15.204103</td>
      <td>5.929589</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.642842e+08</td>
      <td>NaN</td>
      <td>18.688516</td>
      <td>1.494579e+05</td>
      <td>1.616983e+07</td>
      <td>1.612720</td>
      <td>1.181055</td>
    </tr>
  </tbody>
</table>
</div>



```python
from src.genral import save_dataframes_to_pickle
dataframes = [installs, brochure_views, brochure_views_july, app_starts, app_starts_july]
file_names = ['installs.pkl', 'brochure_views.pkl', 'brochure_views_july.pkl', 'app_starts.pkl', 'app_starts_july.pkl']
SAVE_CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data")

save_dataframes_to_pickle(dataframes, file_names, SAVE_CLEAN_DATA_PATH)

```

    File saved successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\clean_data\installs.pkl
    File saved successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\clean_data\brochure_views.pkl
    File saved successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\clean_data\brochure_views_july.pkl
    File saved successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\clean_data\app_starts.pkl
    File saved successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\clean_data\app_starts_july.pkl
    

# Time-sires analysis


```python
earliest_date = brochure_views['dateCreated'].min()
latest_date = brochure_views['dateCreated'].max()

print("Earliest brochure view date:", earliest_date)
print("Latest brochure view date:", latest_date)
```

    Earliest brochure view date: 2017-04-01 00:29:01.711000
    Latest brochure view date: 2017-06-30 23:59:53.699000
    




```python
user_date_range = brochure_views.groupby(['userId','brochure_id']).agg(
    earliest_view=('dateCreated', 'min'),
   latest_view=('dateCreated', 'max'),
    total_duration=('view_duration_log', 'sum'),
    avg_duration=('view_duration_log', 'mean'),
    total_pages=('page_turn_count_log', 'sum')
).reset_index()
user_date_range['view_duration_days'] =np.log1p((user_date_range['latest_view']-user_date_range['earliest_view']).dt.days)

display(user_date_range.head())

print("Number of users with view data:", user_date_range.shape[0])
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
      <th>userId</th>
      <th>brochure_id</th>
      <th>earliest_view</th>
      <th>latest_view</th>
      <th>total_duration</th>
      <th>avg_duration</th>
      <th>total_pages</th>
      <th>view_duration_days</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>541614364</td>
      <td>2017-04-26 03:46:08.280</td>
      <td>2017-04-26 03:46:08.280</td>
      <td>7.601402</td>
      <td>7.601402</td>
      <td>0.693147</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>631731945</td>
      <td>2017-04-24 22:45:25.966</td>
      <td>2017-04-24 22:45:25.966</td>
      <td>11.835016</td>
      <td>11.835016</td>
      <td>3.713572</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>657370331</td>
      <td>2017-04-26 03:46:20.008</td>
      <td>2017-04-26 03:46:20.008</td>
      <td>10.239996</td>
      <td>10.239996</td>
      <td>3.218876</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>665488997</td>
      <td>2017-04-26 03:39:55.547</td>
      <td>2017-04-26 03:39:55.547</td>
      <td>8.853808</td>
      <td>8.853808</td>
      <td>1.098612</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>665489007</td>
      <td>2017-04-28 19:02:13.400</td>
      <td>2017-04-28 19:02:13.400</td>
      <td>8.853808</td>
      <td>8.853808</td>
      <td>1.098612</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
</div>


    Number of users with view data: 197839
    


```python
grouped_by_day = brochure_views.groupby(['userId', 'brochure_id', 'dateCreated']).agg(
    total_views=('view_duration_log', 'count'),
    total_duration=('view_duration_log', 'sum'),
    average_duration=('view_duration_log', 'mean')
).reset_index()
display(grouped_by_day.head())

print("Number of users with view data:", grouped_by_day.shape[0])
print(grouped_by_day['average_duration'].unique().shape[0])

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
      <th>userId</th>
      <th>brochure_id</th>
      <th>dateCreated</th>
      <th>total_views</th>
      <th>total_duration</th>
      <th>average_duration</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>541614364</td>
      <td>2017-04-26 03:46:08.280</td>
      <td>1</td>
      <td>7.601402</td>
      <td>7.601402</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>631731945</td>
      <td>2017-04-24 22:45:25.966</td>
      <td>1</td>
      <td>11.835016</td>
      <td>11.835016</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>657370331</td>
      <td>2017-04-26 03:46:20.008</td>
      <td>1</td>
      <td>10.239996</td>
      <td>10.239996</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>665488997</td>
      <td>2017-04-26 03:39:55.547</td>
      <td>1</td>
      <td>8.853808</td>
      <td>8.853808</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0002C218-D30F-402E-AE08-1280AD4FB669</td>
      <td>665489007</td>
      <td>2017-04-28 19:02:13.400</td>
      <td>1</td>
      <td>8.853808</td>
      <td>8.853808</td>
    </tr>
  </tbody>
</table>
</div>


    Number of users with view data: 279197
    9714
    


```python
plt.figure(figsize=(10,4))
sns.histplot(user_date_range['view_duration_days'], kde=True, bins=30)
plt.title("Distribution of Users view_duration_day ")
plt.xlabel("Date")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](reports/churn%20prediction/02_data_cleaning_after_exploring_files/reports/churn%20prediction/02_data_cleaning_after_exploring_21_0.png)
    



```python
plt.figure(figsize=(10,4))
sns.histplot(user_date_range['earliest_view'], kde=True, bins=30)
plt.title("Distribution of Users' Earliest Brochure View Dates")
plt.xlabel("Date")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](reports/churn%20prediction/02_data_cleaning_after_exploring_files/reports/churn%20prediction/02_data_cleaning_after_exploring_22_0.png)
    



```python
user_date_range['latest_view'] = user_date_range['latest_view'].dt.date

plt.figure(figsize=(10,4))
sns.histplot(user_date_range['latest_view'], kde=True, bins=30, color='orange')
plt.title("Distribution of Users' Latest Brochure View Dates")
plt.xlabel("Date")
plt.ylabel("Number of Users")
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```


    
![png](reports/churn%20prediction/02_data_cleaning_after_exploring_files/reports/churn%20prediction/02_data_cleaning_after_exploring_23_0.png)
    

