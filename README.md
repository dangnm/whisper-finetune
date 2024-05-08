# Fine-tuning and evaluating Whisper models for Automatic Speech Recognition
[Full tutorial here](README_FULL.md)

## Quick example to finetune Whisper model for HASS assist

1. **Edit CSV training data**: 
- Given the sourcWe folder is in C:\ai\whisper-finetune 
- Go to C:\ai\whisper-finetune\custom_data\sample_data and edit the train.csv

2. **Build converted training data**: 
- The output converted train data is in C:\ai\training_data
- Run the following command
```bash
cd C:\ai\whisper-finetune
python custom_data/data_prep.py --source_data_dir C:\ai\whisper-finetune\custom_data\sample_data --output_data_dir C:\ai\training_data
```

3. **Start finetune**: 
- Given the output folder is in C:\ai\output
- Run the following command
```bash
cd C:\ai\whisper-finetune 
python train/fine-tune_on_custom_dataset.py --model_name openai/whisper-base --language Vietnamese --sampling_rate 16000 --num_proc 1 --train_strategy steps --learning_rate 1e-5 --warmup 500 --num_steps 15000 --train_batchsize 8 --eval_batchsize 8 --resume_from_ckpt None --output_dir C:\ai\output --train_datasets C:\ai\training_data --eval_datasets C:\ai\training_data
```

4. **Upload the model to huggingface**: 
- Given the repo Wilber87vn/whisper-base-hass-vn has been created in huggingface
- Given the last checkpoint in C:\ai\output is C:\ai\output\checkpoint-15000
- Create a new folder C:\ai\huggingface_upload
- Copy all files in C:\ai\output\checkpoint-15000 to C:\ai\huggingface_upload
- Download tokenizer.json in [whisper base model](https://huggingface.co/openai/whisper-base/tree/main) and copy it to C:\ai\huggingface_upload
- Run the following code in python

```python
from huggingface_hub import HfApi
api = HfApi()

api.upload_folder(
    folder_path="C:\ai\huggingface_upload",
    repo_id="Wilber87vn/whisper-base-hass-vn",
    repo_type="model",
)
```

5. **Convert to ctranslate2 format**: 
- Install ct2

```bash
pip install ctranslate2
```

- Run the following command

```bash
ct2-transformers-converter --model Wilber87vn/whisper-base-hass-vn --output_dir C:\ai\ct2_output --copy_files tokenizer.json preprocessor_config.json --quantization float16
```
