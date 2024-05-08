# Fine-tuning and evaluating Whisper models for Automatic Speech Recognition
[Full tutorial here](README_FULL.md)

## Quick example to finetune

1. **Edit CSV training data**: 
- Given the source folder is in C:\ai\whisper-finetune 
- Go to C:\ai\whisper-finetune\custom_data\sample_data and edit the train.csv

2. **Build converted training data**: 
- The output converted train data is in C:\ai\training_data

```bash
python custom_data/data_prep.py --source_data_dir C:\ai\whisper-finetune\custom_data\sample_data --output_data_dir C:\ai\training_data
```


