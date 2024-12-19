# label Creation
according to the document a user  is active when it's id is ether in brochure_view or app start

```python
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns

BASE_DIR = os.path.abspath("../../notebooks/reports")
```


```python
# load the clean data set
from src.genral import restore_dataframes_from_pickle
file_names = ['installs.pkl', 'brochure_views.pkl', 'brochure_views_july.pkl', 'app_starts.pkl', 'app_starts_july.pkl']
SAVE_CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "clean_data")
[installs, brochure_views, brochure_views_july, app_starts, app_starts_july] = restore_dataframes_from_pickle(file_names, SAVE_CLEAN_DATA_PATH)
```


```python
active_users_july = set(brochure_views_july['userId'].unique()) | set(app_starts_july['userId'].unique())

```

# feature Engineering
the analysis will go on only userIds in the installed text file


```python
print(brochure_views.shape)
print(app_starts.shape)
brochure_views = brochure_views[brochure_views['userId'].isin(installs['userId'])]
app_starts= app_starts[app_starts['userId'].isin(installs['userId'])]
display(brochure_views.head())
print(brochure_views.shape)
print(app_starts.shape)
```


```python
from datetime import datetime

start_date = '2017-04-01'
end_date = '2017-07-01'
cutoff_date = datetime(2018,7,1)
```


```python
start_dt = pd.to_datetime(start_date)
end_dt = pd.to_datetime(end_date)
```


```python
bv_period = brochure_views[(brochure_views['dateCreated'] >= start_dt) &
                           (brochure_views['dateCreated'] < end_dt)].copy()
as_period = app_starts[(app_starts['dateCreated'] >= start_dt) &
                       (app_starts['dateCreated'] < end_dt)].copy()
display(bv_period.head())
display(as_period.head())
```


```python
events = pd.concat(
    [bv_period[['userId', 'dateCreated']], as_period[['userId', 'dateCreated']]],
    ignore_index=True
)
display(events.head())
print(events.shape)
```


```python

```

## Recency, Frequency, Intensity (RFI)



```python
user_last_eng = events.groupby('userId')['dateCreated'].max().reset_index()
user_last_eng['days_since_last_engagement'] = (cutoff_date - user_last_eng['dateCreated']).dt.days
user_last_eng['days_since_last_engagement'] = user_last_eng['days_since_last_engagement'].clip(lower=0)
display(user_last_eng)
user_last_eng.drop(columns='dateCreated', inplace=True)
```


```python
frequency = events.groupby('userId').agg(total_events=('userId', 'count')).reset_index()
frequency.head()
```


```python
intensity = bv_period.groupby('userId').agg(
        avg_view_duration=('view_duration_log', 'mean'),
        total_page_turns=('page_turn_count_log', 'sum')
    ).reset_index().fillna(0)
intensity.head()
```


```python
features = (user_last_eng.merge(frequency, on='userId', how='left')
                .merge(intensity, on='userId', how='left'))

```


```python
first_interaction = events.groupby('userId')['dateCreated'].min().reset_index()
first_interaction.rename(columns={'dateCreated': 'first_interaction_date'}, inplace=True)
installs = installs.merge(first_interaction, on='userId', how='left')
display(installs.head())
installs['install_to_first_interaction_days'] = (
    (installs['first_interaction_date'] - installs['InstallDate']).dt.days
).clip(lower=0)
mean_install_to_first_interaction = installs['install_to_first_interaction_days'].mean()
installs['install_to_first_interaction_days'] = installs['install_to_first_interaction_days'].fillna(mean_install_to_first_interaction)
features = features.merge(
        installs[['userId', 'install_to_first_interaction_days']],
        on='userId',
        how='left'
)

```


```python
mean_install_to_first_interaction = installs['install_to_first_interaction_days'].mean()
mean_install_to_first_interaction
```


```python
display(features.head())
```

## monthly aggregates



```python
from src.feature_selection import get_months_in_range

months_in_range = get_months_in_range(start_dt, end_dt)
bv_period['year_month'] = bv_period['dateCreated'].dt.to_period('M')

monthly_agg = bv_period.groupby(['userId', 'year_month']).agg(
    monthly_views=('userId', 'count'),
    monthly_avg_duration=('view_duration_log', 'mean'),
    monthly_total_pages=('page_turn_count_log', 'sum')
).reset_index()

monthly_features = monthly_agg.pivot_table(
    index='userId',
    columns='year_month',
    values=['monthly_views', 'monthly_avg_duration', 'monthly_total_pages'],
    fill_value=0
)
```


```python
print(months_in_range)
print(" Monthly Aggregates")
display(monthly_agg.head())
display(monthly_features.head())
```


```python
monthly_features.columns=[''.join([str(c) for c in col]) for col in monthly_features.columns]
print(monthly_features.columns)
```


```python
for ym in months_in_range:
    ym_str = str(ym)
    if f'monthly_views{ym_str}' not in monthly_features.columns:
        monthly_features[f'monthly_views{ym_str}'] = 0
        monthly_features[f'monthly_avg_duration{ym_str}'] = 0
        monthly_features[f'monthly_total_pages{ym_str}'] = 0
```


```python
features = features.merge(monthly_features, on='userId', how='left')
display(features.head())
```

## 3. Active Days & Distinct Brochures


```python
bv_period['view_date'] = bv_period['dateCreated'].dt.date
active_days = bv_period.groupby('userId')['view_date'].nunique().reset_index()
active_days.rename(columns={'view_date': 'active_days'}, inplace=True)

distinct_brochures = bv_period.groupby('userId')['brochure_id'].nunique().reset_index()
distinct_brochures.rename(columns={'brochure_id': 'distinct_brochures'}, inplace=True)

features = features.merge(active_days, on='userId', how='left') \
    .merge(distinct_brochures, on='userId', how='left')
```


```python
display(features.head())
```

# 4. Trend Features (Month-over-Month Differences)



```python
trend_agg = bv_period.groupby(['userId', 'year_month']).agg(
    monthly_views=('userId', 'count')
).reset_index()

# Pivot by year_month
trend_pivot = trend_agg.pivot(index='userId', columns='year_month', values='monthly_views').fillna(0)

# Ensure all months present in the pivot (in case some months had no data)
for ym in months_in_range:
    if ym not in trend_pivot.columns:
        trend_pivot[ym] = 0

# Sort the columns by year_month period to compute differences correctly
sorted_months = sorted(months_in_range)
# Rename columns to views_YYYY-MM
trend_pivot.rename(columns={m: f'views_{m}' for m in trend_pivot.columns}, inplace=True)

# Compute differences between consecutive months
for i in range(1, len(sorted_months)-1):
    curr = sorted_months[i]
    prev = sorted_months[i - 1]
    curr_str = f'views_{curr}'
    prev_str = f'views_{prev}'
    trend_pivot[f'views_diff_{curr}_vs_{prev}'] = trend_pivot[curr_str] - trend_pivot[prev_str]

features = features.merge(trend_pivot, on='userId', how='left')
features = features.fillna(0)
```


```python
display(features.head())
print(features.shape)
```

# Generating Test, Validation, and Training Sets
To perform hyperparameter tuning and model evaluation, we use two different data splits. A third split is reserved for predictions. The data is utilized as follows:

1. **Split 1**:
   - **Training Set**: April
   - **Validation Set**: May

2. **Split 2**:
   - **Training Set**: April and May
   - **Validation Set**: June

3. **Final Prediction Split**:
   - **Training Set**: April, May, and June
   - **Test Set**: July

4. **Deployment Phase**:
   - **Training Set**: April, May, June, and July

## generating the splits




```python

from src.feature_selection import engineer_features_all, generate_active_label
from datetime import datetime
brochure_views = brochure_views[brochure_views['userId'].isin(installs['userId'])]
app_starts= app_starts[app_starts['userId'].isin(installs['userId'])]

# start_date = '2017-04-01'
# end_date = '2017-05-01'
# cutoff_date = datetime(2017,5,1)
# print(brochure_views.shape)
# print(app_starts.shape)
#
# features = engineer_features_all(
#     brochure_views,
#     app_starts,
#     installs,
#     start_date,
#     end_date,
#     cutoff_date
# )
# activity_start='2017-05-01'
# activity_end='2017-06-01'
# label = generate_active_label(
#     brochure_views,
#     app_starts,
#     activity_start,
#     activity_end
# )
# features = features.merge(label, on='userId', how='left')
# features = features.fillna(0)
```


```python
from src.fold_split import generate_splits_and_save
brochure_views = brochure_views[brochure_views['userId'].isin(installs['userId'])]
app_starts= app_starts[app_starts['userId'].isin(installs['userId'])]

SAVE_CLEAN_DATA_PATH = os.path.join(BASE_DIR, "data", "train_set")
brochure_views = brochure_views[brochure_views['userId'].isin(installs['userId'])]
app_starts= app_starts[app_starts['userId'].isin(installs['userId'])]

generate_splits_and_save(
    brochure_views=brochure_views,
    app_starts=app_starts,
    installs=installs,

    base_dir=BASE_DIR,
    save_clean_data_path=SAVE_CLEAN_DATA_PATH,

)

```


```python
from src.feature_selection import engineer_features_all
from src.feature_selection import generate_active_label
train_start_date = '2017-04-01'
train_end_date = '2017-06-30'
train_cutoff_date = pd.to_datetime('2017-07-01')
train_label_start = '2017-06-01'
train_label_end = '2017-06-30'

validation_start_date = '2017-07-01'
validation_end_date = '2017-07-31'
validation_cutoff_date = pd.to_datetime('2017-08-01')
validation_label_start = '2017-07-01'
validation_label_end = '2017-07-31'

train_features = engineer_features_all(
    brochure_views=brochure_views,
    app_starts=app_starts,
    installs=installs,
    start_date=train_start_date,
    end_date=train_end_date,
    cutoff_date=train_cutoff_date
)
train_labels = generate_active_label(
    brochure_views=brochure_views,
    app_starts=app_starts,
    activity_start=train_label_start,
    activity_end=train_label_end
)
train_features = train_features.merge(train_labels, on='userId', how='left')
train_features = train_features.fillna(0)
train_filename = "fold_3_train.pkl"

validation_features = engineer_features_all(
    brochure_views=brochure_views_july,
    app_starts=app_starts_july,
    installs=installs,
    start_date=validation_start_date,
    end_date=validation_end_date,
    cutoff_date=validation_cutoff_date
)
validation_labels = generate_active_label(
    brochure_views=brochure_views_july,
    app_starts=app_starts_july,
    activity_start=validation_label_start,
    activity_end=validation_label_end
)
```


```python
validation_filename = "fold_3_validation.pkl"

```


```python

validation_features = validation_features.merge(train_labels, on='userId', how='left')
validation_features = validation_features.fillna(0)

```


```python

```


```python
from src.genral import save_dataframes_to_pickle
Final_DATA = os.path.join(BASE_DIR, "data", "final_data")
save_dataframes_to_pickle([train_features,validation_features ], [train_filename, validation_filename], Final_DATA)
```
