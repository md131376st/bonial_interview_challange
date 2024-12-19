import pandas as pd
import numpy as np
from datetime import datetime


def get_months_in_range(start_date: datetime, end_date: datetime) -> pd.PeriodIndex:
    start_date = pd.to_datetime("2017-04-01")
    end_date = pd.to_datetime("2017-07-31")
    return pd.date_range(start=start_date, end=end_date, freq='MS').to_period('M')


def engineer_features_all(
        brochure_views: pd.DataFrame,
        app_starts: pd.DataFrame,
        installs: pd.DataFrame,
        start_date: str,
        end_date: str,
        cutoff_date: str  # first day of the month after end_date
) -> pd.DataFrame:
    start_dt = pd.to_datetime(start_date)
    end_dt =  pd.to_datetime(end_date) + pd.Timedelta(days=1) - pd.Timedelta(seconds=1)
    cutoff_date = pd.to_datetime(cutoff_date)

    # Filter to the desired period
    bv_period = brochure_views[(brochure_views['dateCreated'] >= start_dt) &
                               (brochure_views['dateCreated'] < end_dt)].copy()
    as_period = app_starts[(app_starts['dateCreated'] >= start_dt) &
                           (app_starts['dateCreated'] < end_dt)].copy()

    # Combine events for frequency & recency
    events = pd.concat([bv_period[['userId', 'dateCreated']], as_period[['userId', 'dateCreated']]])

    # ------------------------------
    # 1. Recency, Frequency, Intensity (RFI), Install_To_First_Interaction
    # ------------------------------
    # Recency
    user_last_eng = events.groupby('userId')['dateCreated'].max().reset_index()
    user_last_eng['days_since_last_engagement'] = (cutoff_date - user_last_eng['dateCreated']).dt.days
    user_last_eng.drop(columns='dateCreated', inplace=True)
    user_last_eng['days_since_last_engagement'] = user_last_eng['days_since_last_engagement'].clip(lower=0)

    # Frequency
    frequency = (
        events
        .groupby('userId')
        .agg(total_events=('userId', 'count'))
        .reset_index()
    )

    # Intensity
    intensity = bv_period.groupby('userId').agg(
        avg_view_duration=('view_duration_log', 'mean'),
        total_page_turns=('page_turn_count_log', 'sum')
    ).reset_index().fillna(0)

    features = (
        user_last_eng
        .merge(frequency, on='userId', how='left')
        .merge(intensity, on='userId', how='left')
    )
    # Calculate `install_to_first_interaction_days`

    first_interaction = events.groupby('userId')['dateCreated'].min().reset_index()
    first_interaction.rename(columns={'dateCreated': 'first_interaction_date'}, inplace=True)
    install_first_interact = installs.merge(first_interaction, on='userId', how='left')
    install_first_interact['install_to_first_interaction_days'] = (
        (install_first_interact['first_interaction_date'] - install_first_interact['InstallDate']).dt.days
    ).clip(lower=0)
    mean_install_to_first_interaction = install_first_interact['install_to_first_interaction_days'].mean()
    install_first_interact['install_to_first_interaction_days'] = install_first_interact['install_to_first_interaction_days'].fillna(mean_install_to_first_interaction)
    features = features.merge(
        install_first_interact[['userId', 'install_to_first_interaction_days']],
        on='userId',
        how='left'
    )
    # ------------------------------
    # 2. Monthly Aggregates (Multi-Year)
    # ------------------------------
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
    monthly_features.columns = ['{}_{}'.format(col[0], str(col[1])) for col in monthly_features.columns]

    # Add missing months
    for ym in months_in_range:
        ym_str = str(ym)
        for prefix in ['monthly_views', 'monthly_avg_duration', 'monthly_total_pages']:
            col_name = f"{prefix}_{ym_str}"
            if col_name not in monthly_features.columns:
                monthly_features[col_name] = 0


    features = features.merge(monthly_features, on='userId', how='left')

    # ------------------------------
    # 3. Active Days & Distinct Brochures
    # ------------------------------
    bv_period['view_date'] = bv_period['dateCreated'].dt.date
    active_days = bv_period.groupby('userId')['view_date'].nunique().reset_index()
    active_days.rename(columns={'view_date': 'active_days'}, inplace=True)

    distinct_brochures = bv_period.groupby('userId')['brochure_id'].nunique().reset_index()
    distinct_brochures.rename(columns={'brochure_id': 'distinct_brochures'}, inplace=True)

    features = features.merge(active_days, on='userId', how='left') \
        .merge(distinct_brochures, on='userId', how='left')

    # ------------------------------
    # 4. Trend Features (Month-over-Month Differences)
    # ------------------------------

    trend_agg = bv_period.groupby(['userId', 'year_month']).size().unstack(fill_value=0)
    trend_agg.columns = [f"views_{col}" for col in trend_agg.columns]

    # Add missing months for trends
    for ym in months_in_range:
        col_name = f"views_{ym}"
        if col_name not in trend_agg.columns:
            trend_agg[col_name] = 0

    trend_agg = trend_agg.sort_index(axis=1)  # Ensure months are in chronological order
    for i in range(1, len(months_in_range)):
        curr = months_in_range[i]
        prev = months_in_range[i - 1]
        trend_agg[f"views_diff_{curr}_vs_{prev}"] = trend_agg[f"views_{curr}"] - trend_agg[f"views_{prev}"]

    features = features.merge(trend_agg, on='userId', how='left')
    features = features.fillna(0)

    return features


def generate_active_label(
        brochure_views: pd.DataFrame,
        app_starts: pd.DataFrame,
        activity_start: str,
        activity_end: str
) -> pd.DataFrame:

    activity_start_dt = pd.to_datetime(activity_start)
    activity_end_dt = pd.to_datetime(activity_end)

    active_users_bv = brochure_views[(brochure_views['dateCreated'] >= activity_start_dt) &
                                     (brochure_views['dateCreated'] <= activity_end_dt)]['userId']
    active_users_as = app_starts[(app_starts['dateCreated'] > activity_start_dt) &
                                 (app_starts['dateCreated'] < activity_end_dt)]['userId']

    active_users = set(active_users_bv.unique()) | set(active_users_as.unique())

    is_active = pd.DataFrame({'userId': list(active_users), 'is_active': 1})

    return is_active