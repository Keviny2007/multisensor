"""
Preprocessing script to merge 4-sensor accelerometer data into single CSV format.

Input format (per subject):
- P00X_RW.csv: Right Wrist [timestamp, x, y, z]
- P00X_LW.csv: Left Wrist [timestamp, x, y, z]
- P00X_RL.csv: Right Leg [timestamp, x, y, z]
- P00X_LL.csv: Left Leg [timestamp, x, y, z]
- P00X_ground_truth (apple watch).csv: Labels [timestamp, sleep_stage]

Output format:
- P00X.csv: [timestamp, RW_x, RW_y, RW_z, LW_x, LW_y, LW_z, RL_x, RL_y, RL_z, LL_x, LL_y, LL_z, label]
"""

import os
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm


def load_sensor_data(sensor_path):
    """Load a single sensor CSV file.

    Parameters
    ----------
    sensor_path : str or Path
        Path to sensor CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [timestamp, x, y, z]
    """
    df = pd.read_csv(sensor_path, header=None, names=['timestamp', 'x', 'y', 'z'])
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def load_ground_truth(gt_path):
    """Load ground truth labels file.

    Parameters
    ----------
    gt_path : str or Path
        Path to ground truth CSV file

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: [timestamp, sleep_stage]
    """
    df = pd.read_csv(gt_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df


def align_sensors(rw_df, lw_df, rl_df, ll_df):
    """Align 4 sensors by timestamp using nearest neighbor matching.

    Uses Right Wrist as the reference timestamp and aligns others to it.

    Parameters
    ----------
    rw_df, lw_df, rl_df, ll_df : pd.DataFrame
        DataFrames for each sensor

    Returns
    -------
    pd.DataFrame
        Merged DataFrame with aligned timestamps
    """
    # Use Right Wrist as reference
    merged = rw_df.copy()
    merged.columns = ['timestamp', 'RW_x', 'RW_y', 'RW_z']

    # Merge each sensor using nearest timestamp (within 50ms tolerance)
    for sensor_df, prefix in [(lw_df, 'LW'), (rl_df, 'RL'), (ll_df, 'LL')]:
        sensor_df = sensor_df.copy()
        sensor_df.columns = ['timestamp', f'{prefix}_x', f'{prefix}_y', f'{prefix}_z']

        # Use merge_asof for nearest timestamp matching
        merged = pd.merge_asof(
            merged.sort_values('timestamp'),
            sensor_df.sort_values('timestamp'),
            on='timestamp',
            direction='nearest',
            tolerance=pd.Timedelta('50ms')
        )

    return merged


def upsample_labels(sensor_df, gt_df):
    """Upsample ground truth labels from 30s epochs to sensor sampling rate.

    Parameters
    ----------
    sensor_df : pd.DataFrame
        Sensor data with high-frequency timestamps
    gt_df : pd.DataFrame
        Ground truth labels at 30s intervals

    Returns
    -------
    pd.DataFrame
        Sensor data with upsampled labels
    """
    # Merge labels using forward fill
    merged = pd.merge_asof(
        sensor_df.sort_values('timestamp'),
        gt_df.sort_values('timestamp'),
        on='timestamp',
        direction='backward'
    )

    return merged


def process_subject(subject_dir, output_dir, label_map=None):
    """Process a single subject's data.

    Parameters
    ----------
    subject_dir : str or Path
        Path to subject directory containing sensor CSVs
    output_dir : str or Path
        Directory to save processed CSV
    label_map : dict, optional
        Mapping of label strings to integers (e.g., {'Wake': 0, 'Sleep': 1})

    Returns
    -------
    bool
        True if successful, False otherwise
    """
    subject_dir = Path(subject_dir)
    subject_id = subject_dir.name.split('_')[0]  # Extract P003 from "P003_251031 (dylan)"

    print(f"\nProcessing {subject_id}...")

    # Find sensor files
    sensor_files = {
        'RW': None,
        'LW': None,
        'RL': None,
        'LL': None,
        'GT': None
    }

    for file in subject_dir.iterdir():
        if file.suffix == '.csv':
            if 'RW' in file.name:
                sensor_files['RW'] = file
            elif 'LW' in file.name:
                sensor_files['LW'] = file
            elif 'RL' in file.name:
                sensor_files['RL'] = file
            elif 'LL' in file.name:
                sensor_files['LL'] = file
            elif 'ground_truth' in file.name:
                sensor_files['GT'] = file

    # Check if all files exist
    missing = [k for k, v in sensor_files.items() if v is None]
    if missing:
        print(f"  ERROR: Missing files for {subject_id}: {missing}")
        return False

    try:
        # Load sensor data
        print(f"  Loading sensor data...")
        rw_df = load_sensor_data(sensor_files['RW'])
        lw_df = load_sensor_data(sensor_files['LW'])
        rl_df = load_sensor_data(sensor_files['RL'])
        ll_df = load_sensor_data(sensor_files['LL'])

        print(f"    RW: {len(rw_df)} samples")
        print(f"    LW: {len(lw_df)} samples")
        print(f"    RL: {len(rl_df)} samples")
        print(f"    LL: {len(ll_df)} samples")

        # Load ground truth
        print(f"  Loading ground truth...")
        gt_df = load_ground_truth(sensor_files['GT'])
        print(f"    Labels: {len(gt_df)} epochs (30s each)")
        print(f"    Label distribution: {gt_df['sleep_stage'].value_counts().to_dict()}")

        # Align sensors
        print(f"  Aligning sensors by timestamp...")
        merged_df = align_sensors(rw_df, lw_df, rl_df, ll_df)
        print(f"    Merged: {len(merged_df)} samples")

        # Check for missing values
        missing_count = merged_df.isnull().sum().sum()
        if missing_count > 0:
            print(f"    WARNING: {missing_count} missing values after alignment")
            print(f"    Dropping rows with missing values...")
            merged_df = merged_df.dropna()
            print(f"    After dropping: {len(merged_df)} samples")

        # Upsample labels
        print(f"  Upsampling labels to sensor frequency...")
        final_df = upsample_labels(merged_df, gt_df)

        # Map labels to integers if mapping provided
        if label_map:
            final_df['label'] = final_df['sleep_stage'].map(label_map)
            print(f"    Mapped labels: {final_df['label'].value_counts().to_dict()}")
        else:
            final_df['label'] = final_df['sleep_stage']

        # Check for unlabeled data
        unlabeled = final_df['label'].isnull().sum()
        if unlabeled > 0:
            print(f"    WARNING: {unlabeled} samples without labels (dropping)")
            final_df = final_df.dropna(subset=['label'])

        # Select final columns
        final_columns = [
            'timestamp',
            'RW_x', 'RW_y', 'RW_z',
            'LW_x', 'LW_y', 'LW_z',
            'RL_x', 'RL_y', 'RL_z',
            'LL_x', 'LL_y', 'LL_z',
            'label'
        ]
        final_df = final_df[final_columns]

        # Save processed data
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        output_path = output_dir / f"{subject_id}.csv"

        print(f"  Saving to {output_path}...")
        final_df.to_csv(output_path, index=False)

        # Print summary
        print(f"  ✓ Success!")
        print(f"    Output: {len(final_df)} samples")
        print(f"    Duration: {(final_df['timestamp'].max() - final_df['timestamp'].min()).total_seconds() / 3600:.2f} hours")
        print(f"    Sampling rate: ~{len(final_df) / (final_df['timestamp'].max() - final_df['timestamp'].min()).total_seconds():.1f} Hz")

        return True

    except Exception as e:
        print(f"  ERROR processing {subject_id}: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main preprocessing function."""
    # Configuration
    input_dir = Path("data/test_data")
    output_dir = Path("data/processed_test_data")

    # Label mapping: Wake=0, Sleep=1 (binary classification)
    label_map = {
        'Wake': 0,
        'Sleep': 1
    }

    print("=" * 60)
    print("4-Sensor Data Preprocessing")
    print("=" * 60)
    print(f"Input directory: {input_dir}")
    print(f"Output directory: {output_dir}")
    print(f"Label mapping: {label_map}")
    print()

    # Find all subject directories
    subject_dirs = [d for d in input_dir.iterdir() if d.is_dir()]
    print(f"Found {len(subject_dirs)} subjects")

    # Process each subject
    results = {}
    for subject_dir in sorted(subject_dirs):
        success = process_subject(subject_dir, output_dir, label_map)
        results[subject_dir.name] = success

    # Print summary
    print("\n" + "=" * 60)
    print("Processing Summary")
    print("=" * 60)
    successful = sum(results.values())
    total = len(results)
    print(f"Successful: {successful}/{total}")
    print()

    for subject, success in results.items():
        status = "✓" if success else "✗"
        print(f"  {status} {subject}")

    if successful == total:
        print("\n✓ All subjects processed successfully!")
        print(f"\nProcessed data saved to: {output_dir}")
    else:
        print("\n⚠ Some subjects failed to process. Check errors above.")


if __name__ == "__main__":
    main()
