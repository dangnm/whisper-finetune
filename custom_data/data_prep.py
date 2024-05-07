# python custom_data/data_prep.py --source_data_dir source_data_directory  --output_data_dir output_data_directory
import argparse
from datasets import Dataset, Audio, Value
import pandas as pd

parser = argparse.ArgumentParser(description='Preliminary data preparation script before Whisper Fine-tuning.')
parser.add_argument('--source_data_dir', type=str, required=True, default=False, help='Path to the directory containing the audio_paths and text files.')
parser.add_argument('--output_data_dir', type=str, required=False, default='op_data_dir', help='Output data directory path.')

args = parser.parse_args()

train_df = pd.read_csv(f"{args.source_data_dir}/train.csv")
train_df.columns = ["audio", "sentence"]
train_dataset = Dataset.from_pandas(train_df)
train_dataset = train_dataset.cast_column("audio", Audio(sampling_rate=16000))
train_dataset = train_dataset.cast_column("sentence", Value("string"))
train_dataset.save_to_disk(args.output_data_dir)
print('Data preparation done')
