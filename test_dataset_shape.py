"""Quick test to check dataset shapes"""
import sys
sys.path.insert(0, '.')

import yaml
import src.datasets
import src.config

# Load config
config_path = 'params/MultiPair_downstream_experiments/MultiPair_FourSensor_bilateral_linear_probe/config_test.yml'
with open(config_path, 'r') as f:
    cfg_dict = yaml.safe_load(f)

# Add CONFIG_PATH
cfg_dict['CONFIG_PATH'] = 'params/MultiPair_downstream_experiments/MultiPair_FourSensor_bilateral_linear_probe'

config = src.config.Config(cfg_dict)

# Create dataset
print("Creating dataset...")

# Get dataset args (handle both list and dict)
ds_args = config.DATASET_ARGS
if isinstance(ds_args, list):
    ds_args = ds_args[0]

# Unwrap list values (config uses lists for grid search)
ds_args_unwrapped = {}
for k, v in ds_args.items():
    if isinstance(v, list) and len(v) == 1:
        ds_args_unwrapped[k] = v[0]
    else:
        ds_args_unwrapped[k] = v
ds_args = ds_args_unwrapped

dataset = src.datasets.get_dataset(
    dataset_name=config.DATASET,
    dataset_args=ds_args,
    root_dir=config.TRAIN_DATA,
    num_classes=config.num_classes,
    label_map=config.label_index,
    replace_classes=config.replace_classes,
    config_path=config.CONFIG_PATH,
    skip_files=['P005.csv'],
    name_label_map=config.class_name_label_map
)

print(f"\nDataset info:")
print(f"  Total size: {len(dataset)}")
print(f"  Feature dim: {dataset.feature_dim}")
print(f"  Sequence length: {dataset.seq_length}")

# Get first item
print(f"\nGetting first item...")
pair1, pair2, y = dataset[0]

print(f"\nShapes:")
print(f"  pair1: {pair1.shape}")
print(f"  pair2: {pair2.shape}")
print(f"  y: {y.shape}")

print(f"\nExpected pair shapes: [seq_len, 9006]")
print(f"  where 9006 = (n_fft//2+1) * 6_channels = 1501 * 6")
