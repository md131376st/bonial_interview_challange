import os
import pandas as pd

from src.feature_selection import engineer_features_all, generate_active_label


def generate_splits_and_save(
        brochure_views: pd.DataFrame,
        app_starts: pd.DataFrame,
        installs:pd.DataFrame,
        base_dir: str,
        save_clean_data_path: str
):
    splits = {
        # Split 1: April -> May
        'fold_1': {
        'train': {
            'start_date': '2017-04-01',
            'end_date': '2017-04-30',  # Features from April
            'cutoff_date': '2017-05-01',  # Start of May
            'label_start': '2017-05-01',  # Labels from May activity
            'label_end': '2017-05-31'
        },
        'validation': {
            'start_date': '2017-05-01',
            'end_date': '2017-05-31',  # Features from May
            'cutoff_date': '2017-06-01',  # Start of June
            'label_start': '2017-05-01',  # Labels from May activity
            'label_end': '2017-05-31'
        }
    },
    # Fold 2: Train April-May -> Validate June
    'fold_2': {
        'train': {
            'start_date': '2017-04-01',
            'end_date': '2017-05-31',  # Features from April and May
            'cutoff_date': '2017-06-01',  # Start of June
            'label_start': '2017-06-01',  # Labels from June activity
            'label_end': '2017-06-30'
        },
        'validation': {
            'start_date': '2017-06-01',
            'end_date': '2017-06-30',  # Features from June
            'cutoff_date': '2017-07-01',  # Start of July
            'label_start': '2017-06-01',  # Labels from June activity
            'label_end': '2017-06-30'
        }
    }
    }

    # Ensure save path exists
    save_path = os.path.join(base_dir, save_clean_data_path)
    os.makedirs(save_path, exist_ok=True)

    for split_name, split_data in splits.items():
        print(f"Processing {split_name.upper()}:")

        # Training Data
        train = split_data.get('train')
        if train:
            print(f"  Training Set: {train['start_date']} to {train['end_date']}")

            train_features = engineer_features_all(
                brochure_views=brochure_views,
                app_starts=app_starts,
                installs=installs,
                start_date=train['start_date'],
                end_date=train['end_date'],
                cutoff_date=train['cutoff_date']
            )
            train_labels = generate_active_label(
                brochure_views=brochure_views,
                app_starts=app_starts,
                activity_start=train['label_start'],
                activity_end=train['label_end']
            )
            train_features = train_features.merge(train_labels, on='userId', how='left')
            train_features = train_features.fillna(0)

            train_filename = f"{split_name}_train.pkl"
            train_features.to_pickle(os.path.join(save_path, train_filename))
            print(f"Saved Training Features: {train_filename}")

        validation_or_test = split_data.get('validation') or split_data.get('test')
        if validation_or_test:
            print(f"  Validation/Test Set: {validation_or_test['start_date']} to {validation_or_test['end_date']}")

            val_test_features = engineer_features_all(
                brochure_views=brochure_views,
                app_starts=app_starts,
                installs=installs,
                start_date=validation_or_test['start_date'],
                end_date=validation_or_test['end_date'],
                cutoff_date=validation_or_test['cutoff_date']
            )

            val_test_labels = generate_active_label(
                brochure_views=brochure_views,
                app_starts=app_starts,
                activity_start=validation_or_test['label_start'],
                activity_end=validation_or_test['label_end']
            )

            val_test_features = val_test_features.merge(val_test_labels, on='userId', how='left')
            val_test_features = val_test_features.fillna(0)

            val_test_filename = f"{split_name}_validation_or_test.pkl"
            val_test_features.to_pickle(os.path.join(save_path, val_test_filename))
            print(f"Saved Validation/Test Features: {val_test_filename}")

    print("All splits processed and saved successfully!")
