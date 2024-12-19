```python
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score
import os
```


```python
BASE_DIR = os.path.abspath("..")
TRAIN_SET = os.path.join(BASE_DIR,"data", "train_set")

```


```python
from src.genral import restore_dataframes_from_pickle

[train_set1, validation_set1, train_set2, validation_set2] = restore_dataframes_from_pickle(
    file_names=["fold_1_train.pkl", "fold_1_validation_or_test.pkl", "fold_2_train.pkl", "fold_2_validation_or_test.pkl"],
    folder_path=TRAIN_SET
)

display(train_set1.head())
display(validation_set1.head())
display(train_set2.head())
display(validation_set2.head())
```

    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\train_set\fold_1_train.pkl
    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\train_set\fold_1_validation_or_test.pkl
    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\train_set\fold_2_train.pkl
    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\train_set\fold_2_validation_or_test.pkl
    


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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-04</th>
      <th>monthly_total_pages_2017-04</th>
      <th>monthly_views_2017-04</th>
      <th>monthly_views_2017-05</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002c218-d30f-402e-ae08-1280ad4fb669</td>
      <td>1</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000691c6-4289-47f8-81f1-628e52ed5429</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00095350-9e64-4b34-9112-b9869703248b</td>
      <td>22</td>
      <td>7</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>0.0</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00144cb3-bd42-48a3-bae7-53d58e509a3b</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-05</th>
      <th>monthly_total_pages_2017-05</th>
      <th>monthly_views_2017-05</th>
      <th>monthly_views_2017-04</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000bb01-a52b-4b4c-a0dd-8ef80f0a810c</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>22</td>
      <td>2</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00144cb3-bd42-48a3-bae7-53d58e509a3b</td>
      <td>21</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0015a4a8-99f1-4119-9e10-0ac9773ae48a</td>
      <td>23</td>
      <td>4</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>0.0</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-04</th>
      <th>monthly_avg_duration_2017-05</th>
      <th>monthly_total_pages_2017-04</th>
      <th>monthly_total_pages_2017-05</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000bb01-a52b-4b4c-a0dd-8ef80f0a810c</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002c218-d30f-402e-ae08-1280ad4fb669</td>
      <td>32</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000691c6-4289-47f8-81f1-628e52ed5429</td>
      <td>36</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00095350-9e64-4b34-9112-b9869703248b</td>
      <td>53</td>
      <td>7</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>0.0</td>
      <td>10.807466</td>
      <td>0.000000</td>
      <td>14.632555</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>22</td>
      <td>2</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>9.546884</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-06</th>
      <th>monthly_total_pages_2017-06</th>
      <th>monthly_views_2017-06</th>
      <th>monthly_views_2017-04</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>9</td>
      <td>6</td>
      <td>10.523636</td>
      <td>5.662960</td>
      <td>33.0</td>
      <td>10.523636</td>
      <td>5.662960</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>53</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>43.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001c5296-292d-4290-a1ac-b5751eac80c2</td>
      <td>11</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0020c0ac-da8c-460c-91ce-b1b603456180</td>
      <td>1</td>
      <td>125</td>
      <td>11.013475</td>
      <td>197.345937</td>
      <td>3.0</td>
      <td>11.013475</td>
      <td>197.345937</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>-87.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021b2c0-a153-4da8-bf33-56fb39a83525</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



```python
# Train and Validation sets for Fold 1
X_train_fold1 = train_set1.drop(columns=["userId", "is_active"])
y_train_fold1 = train_set1["is_active"]
X_val_fold1 = validation_set1.drop(columns=["userId", "is_active"])
y_val_fold1 = validation_set1["is_active"]

# Train and Validation sets for Fold 2
X_train_fold2 = train_set2.drop(columns=["userId", "is_active"])
y_train_fold2 = train_set2["is_active"]
X_val_fold2 = validation_set2.drop(columns=["userId", "is_active"])
y_val_fold2 = validation_set2["is_active"]
```


```python
display(train_set1.head())
display(validation_set1.head())
display(train_set2.head())
display(validation_set2.head())
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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-04</th>
      <th>monthly_total_pages_2017-04</th>
      <th>monthly_views_2017-04</th>
      <th>monthly_views_2017-05</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0002c218-d30f-402e-ae08-1280ad4fb669</td>
      <td>1</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>000691c6-4289-47f8-81f1-628e52ed5429</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>00095350-9e64-4b34-9112-b9869703248b</td>
      <td>22</td>
      <td>7</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>0.0</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00144cb3-bd42-48a3-bae7-53d58e509a3b</td>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-05</th>
      <th>monthly_total_pages_2017-05</th>
      <th>monthly_views_2017-05</th>
      <th>monthly_views_2017-04</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000bb01-a52b-4b4c-a0dd-8ef80f0a810c</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>22</td>
      <td>2</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00144cb3-bd42-48a3-bae7-53d58e509a3b</td>
      <td>21</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0015a4a8-99f1-4119-9e10-0ac9773ae48a</td>
      <td>23</td>
      <td>4</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>0.0</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-04</th>
      <th>monthly_avg_duration_2017-05</th>
      <th>monthly_total_pages_2017-04</th>
      <th>monthly_total_pages_2017-05</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0000bb01-a52b-4b4c-a0dd-8ef80f0a810c</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0002c218-d30f-402e-ae08-1280ad4fb669</td>
      <td>32</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>000691c6-4289-47f8-81f1-628e52ed5429</td>
      <td>36</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>00095350-9e64-4b34-9112-b9869703248b</td>
      <td>53</td>
      <td>7</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>0.0</td>
      <td>10.807466</td>
      <td>0.000000</td>
      <td>14.632555</td>
      <td>0.000000</td>
      <td>...</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>22</td>
      <td>2</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>9.546884</td>
      <td>0.000000</td>
      <td>0.693147</td>
      <td>...</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



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
      <th>days_since_last_engagement</th>
      <th>total_events</th>
      <th>avg_view_duration</th>
      <th>total_page_turns</th>
      <th>install_to_first_interaction_days</th>
      <th>monthly_avg_duration_2017-06</th>
      <th>monthly_total_pages_2017-06</th>
      <th>monthly_views_2017-06</th>
      <th>monthly_views_2017-04</th>
      <th>...</th>
      <th>active_days</th>
      <th>distinct_brochures</th>
      <th>views_2017-04</th>
      <th>views_2017-05</th>
      <th>views_2017-06</th>
      <th>views_2017-07</th>
      <th>views_diff_2017-05_vs_2017-04</th>
      <th>views_diff_2017-06_vs_2017-05</th>
      <th>views_diff_2017-07_vs_2017-06</th>
      <th>is_active</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>00095653-80f8-4fba-93d9-44ae70bb6263</td>
      <td>9</td>
      <td>6</td>
      <td>10.523636</td>
      <td>5.662960</td>
      <td>33.0</td>
      <td>10.523636</td>
      <td>5.662960</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>0010e3be-81bd-48a3-8282-8c8d0b1f9629</td>
      <td>0</td>
      <td>53</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>43.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>001c5296-292d-4290-a1ac-b5751eac80c2</td>
      <td>11</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>22.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0020c0ac-da8c-460c-91ce-b1b603456180</td>
      <td>1</td>
      <td>125</td>
      <td>11.013475</td>
      <td>197.345937</td>
      <td>3.0</td>
      <td>11.013475</td>
      <td>197.345937</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>24.0</td>
      <td>70.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>87.0</td>
      <td>-87.0</td>
      <td>1.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>0021b2c0-a153-4da8-bf33-56fb39a83525</td>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>36.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 28 columns</p>
</div>



```python
print(X_train_fold1.shape)
print(X_val_fold1.shape)
print(X_train_fold2.shape)
print(X_val_fold2.shape)
X_train_fold1.columns=range(X_train_fold1.shape[1])
X_val_fold1.columns=range(X_train_fold1.shape[1])
X_train_fold2.columns=range(X_train_fold1.shape[1])
X_val_fold2.columns=range(X_train_fold1.shape[1])
```

    (9801, 26)
    (13609, 26)
    (19921, 26)
    (6167, 26)
    


```python
X_train_fold1.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1</td>
      <td>9</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>22</td>
      <td>7</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>0.0</td>
      <td>10.807466</td>
      <td>14.632555</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>2.0</td>
      <td>5.0</td>
      <td>5.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>-5.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>0</td>
      <td>19</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>5</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>



# list of models to train


```python
from lightgbm import LGBMClassifier
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
# from xgboost import XGBClassifier
# from lightgbm import LGBMClassifier
models = {
    "Logistic Regression": {
        "model": LogisticRegression(class_weight="balanced", random_state=42),
        "params": {
            "penalty": ["l2"],
            "C": [0.01, 0.1, 1, 10],
            "solver": ["liblinear", "lbfgs"]
        }
    },
    "Random Forest": {
        "model": RandomForestClassifier(class_weight="balanced", random_state=42),
        "params": {
            "n_estimators": [50, 100, 200],
            "max_depth": [5, 10, None],
            "min_samples_split": [2, 5, 10]
        }
    },
    "XGBoost": {
        "model": XGBClassifier(random_state=42, use_label_encoder=False, eval_metric="logloss"),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [3, 5, 10],
            "learning_rate": [0.01, 0.1, 0.2],
            "scale_pos_weight": [1, 2, 5]
        }
    },
    "LightGBM": {
        "model": LGBMClassifier(class_weight="balanced", random_state=42),
        "params": {
            "n_estimators": [50, 100],
            "max_depth": [5, 10, -1],
            "learning_rate": [0.01, 0.1, 0.2]
        }
    }
}



```


```python
X_train_fold1.head()
y_train_fold1.head()
X_val_fold1.head()
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
      <th>0</th>
      <th>1</th>
      <th>2</th>
      <th>3</th>
      <th>4</th>
      <th>5</th>
      <th>6</th>
      <th>7</th>
      <th>8</th>
      <th>9</th>
      <th>...</th>
      <th>16</th>
      <th>17</th>
      <th>18</th>
      <th>19</th>
      <th>20</th>
      <th>21</th>
      <th>22</th>
      <th>23</th>
      <th>24</th>
      <th>25</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>23</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>1</th>
      <td>22</td>
      <td>2</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>0.0</td>
      <td>9.546884</td>
      <td>0.693147</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>-1.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>2</th>
      <td>0</td>
      <td>49</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>13.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>3</th>
      <td>21</td>
      <td>1</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>14.0</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>0.0</td>
    </tr>
    <tr>
      <th>4</th>
      <td>23</td>
      <td>4</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>0.0</td>
      <td>10.145755</td>
      <td>4.836282</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>...</td>
      <td>0.0</td>
      <td>1.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>0.0</td>
      <td>0.0</td>
      <td>3.0</td>
      <td>-3.0</td>
      <td>0.0</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 26 columns</p>
</div>




```python
from src.Models import train_and_evaluate

best_model_fold1, best_model_name_fold1, best_params_fold1, best_score_fold1 = train_and_evaluate(
    X_train_fold1, y_train_fold1, X_val_fold1, y_val_fold1, models
)
```

    Training Logistic Regression...
    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    Logistic Regression ROC-AUC on Validation: 0.3705
    Training Random Forest...
    Fitting 5 folds for each of 27 candidates, totalling 135 fits
    Random Forest ROC-AUC on Validation: 0.3222
    Training XGBoost...
    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\xgboost\core.py:158: UserWarning: [18:00:45] WARNING: D:\bld\xgboost-split_1733179535861\work\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

    XGBoost ROC-AUC on Validation: 0.3171
    Training LightGBM...
    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    [LightGBM] [Info] Number of positive: 3462, number of negative: 6339
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000085 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 1501
    [LightGBM] [Info] Number of data points in the train set: 9801, number of used features: 11
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=-0.000000
    [LightGBM] [Info] Start training from score -0.000000
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    [LightGBM] [Warning] No further splits with positive gain, best gain: -inf
    LightGBM ROC-AUC on Validation: 0.3108
    
    Best Model: Logistic Regression
    Best Params: {'C': 0.1, 'penalty': 'l2', 'solver': 'lbfgs'}
    Best ROC-AUC: 0.3705
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,
    


```python
best_model_fold2, best_model_name_fold2, best_params_fold2, best_score_fold2 = train_and_evaluate(
    X_train_fold2, y_train_fold2, X_val_fold2, y_val_fold2, models
)
```

    Training Logistic Regression...
    Fitting 5 folds for each of 8 candidates, totalling 40 fits
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\sklearn\linear_model\_logistic.py:469: ConvergenceWarning: lbfgs failed to converge (status=1):
    STOP: TOTAL NO. of ITERATIONS REACHED LIMIT.
    
    Increase the number of iterations (max_iter) or scale the data as shown in:
        https://scikit-learn.org/stable/modules/preprocessing.html
    Please also refer to the documentation for alternative solver options:
        https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression
      n_iter_i = _check_optimize_result(
    

    Logistic Regression ROC-AUC on Validation: 0.5567
    Training Random Forest...
    Fitting 5 folds for each of 27 candidates, totalling 135 fits
    Random Forest ROC-AUC on Validation: 0.1214
    Training XGBoost...
    Fitting 5 folds for each of 54 candidates, totalling 270 fits
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\xgboost\core.py:158: UserWarning: [18:01:38] WARNING: D:\bld\xgboost-split_1733179535861\work\src\learner.cc:740: 
    Parameters: { "use_label_encoder" } are not used.
    
      warnings.warn(smsg, UserWarning)
    

    XGBoost ROC-AUC on Validation: 0.6096
    Training LightGBM...
    Fitting 5 folds for each of 18 candidates, totalling 90 fits
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,
    

    [LightGBM] [Info] Number of positive: 6118, number of negative: 13803
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000931 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 2598
    [LightGBM] [Info] Number of data points in the train set: 19921, number of used features: 17
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    LightGBM ROC-AUC on Validation: 0.6472
    
    Best Model: LightGBM
    Best Params: {'learning_rate': 0.01, 'max_depth': -1, 'n_estimators': 100}
    Best ROC-AUC: 0.6472
    


```python
Final_DATA = os.path.join(BASE_DIR, "data", "final_data")
[train_set1, validation_set1] = restore_dataframes_from_pickle(
    file_names=["fold_3_train.pkl", "fold_3_validation.pkl"],
    folder_path=Final_DATA
)
X_train_fold1 = train_set1.drop(columns=["userId", "is_active"])
y_train_fold1 = train_set1["is_active"]
X_val_fold1 = validation_set1.drop(columns=["userId", "is_active"])
y_val_fold1 = validation_set1["is_active"]
X_train_fold1.columns=range(X_train_fold1.shape[1])
X_val_fold1.columns=range(X_train_fold1.shape[1])
models = {
    "LightGBM": {
        "model": LGBMClassifier(class_weight="balanced", random_state=42),
        "params": {
            "n_estimators": [100],
            "max_depth": [-1],
            "learning_rate": [0.01]
        }
    }
}
train_and_evaluate(
    X_train_fold2, y_train_fold2, X_val_fold2, y_val_fold2, models
)

```

    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\final_data\fold_3_train.pkl
    File restored successfully: C:\Users\mona1\PycharmProjects\scientificProject\data\final_data\fold_3_validation.pkl
    Training LightGBM...
    Fitting 5 folds for each of 1 candidates, totalling 5 fits
    

    C:\Users\mona1\Desktop\Bonial\Lib\site-packages\numpy\ma\core.py:2881: RuntimeWarning: invalid value encountered in cast
      _data = np.array(data, dtype=dtype, copy=copy,
    

    [LightGBM] [Info] Number of positive: 6118, number of negative: 13803
    [LightGBM] [Info] Auto-choosing row-wise multi-threading, the overhead of testing was 0.000773 seconds.
    You can set `force_row_wise=true` to remove the overhead.
    And if memory is not enough, you can set `force_col_wise=true`.
    [LightGBM] [Info] Total Bins 2598
    [LightGBM] [Info] Number of data points in the train set: 19921, number of used features: 17
    [LightGBM] [Info] [binary:BoostFromScore]: pavg=0.500000 -> initscore=0.000000
    [LightGBM] [Info] Start training from score 0.000000
    LightGBM ROC-AUC on Validation: 0.6472
    
    Best Model: LightGBM
    Best Params: {'learning_rate': 0.01, 'max_depth': -1, 'n_estimators': 100}
    Best ROC-AUC: 0.6472
    




    (LGBMClassifier(class_weight='balanced', learning_rate=0.01, random_state=42),
     'LightGBM',
     {'learning_rate': 0.01, 'max_depth': -1, 'n_estimators': 100},
     np.float64(0.6472342079689019))


