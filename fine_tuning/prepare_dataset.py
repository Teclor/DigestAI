import os
from datasets import load_dataset
import pandas as pd
from transformers import AutoTokenizer
import random

MAX_SOURCE_LENGTH = 512
MODEL_NAME = "unsloth/gemma-3-1b-it-qat"
SAMPLE_SIZE = 75000  # количество случайных примеров в итоговом датасете

def save_dataset_as_parquet():
    print("Загрузка датасета...")
    dataset = load_dataset("RussianNLP/Mixed-Summarization-Dataset")

    print(f"Загрузка токенизатора {MODEL_NAME}...")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)
    tokenizer.pad_token = tokenizer.eos_token

    def filter_by_length(example):
        tokens = tokenizer.tokenize(example["text"])
        return len(tokens) <= MAX_SOURCE_LENGTH

    print(f"Фильтрация train датасета по длине текста <= {MAX_SOURCE_LENGTH} токенов...")
    filtered_train = dataset["train"].filter(filter_by_length, num_proc=14, desc="Filtering train")
    print(f"Фильтрация test датасета по длине текста <= {MAX_SOURCE_LENGTH} токенов...")
    filtered_test = dataset["test"].filter(filter_by_length, num_proc=14, desc="Filtering test")

    print(f"Осталось примеров в train после фильтрации: {len(filtered_train)}")
    print(f"Осталось примеров в test после фильтрации: {len(filtered_test)}")

    def shuffle_and_sample(ds, size):
        print(f"Перемешивание и выборка {size} случайных примеров...")
        indices = list(range(len(ds)))
        random.shuffle(indices)
        selected_indices = indices[:min(size, len(ds))]
        return ds.select(selected_indices)

    filtered_train = shuffle_and_sample(filtered_train, SAMPLE_SIZE)
    filtered_test = shuffle_and_sample(filtered_test, SAMPLE_SIZE)

    print(f"Итоговое количество примеров в train: {len(filtered_train)}")
    print(f"Итоговое количество примеров в test: {len(filtered_test)}")

    print("Конвертация в pandas DataFrame...")
    train_df = pd.DataFrame(filtered_train)
    test_df = pd.DataFrame(filtered_test)

    os.makedirs("resources", exist_ok=True)

    print("Сохранение train датасета в resources/summarization_ds_train.parquet...")
    train_df.to_parquet("resources/summarization_ds_train.parquet", index=False)

    print("Сохранение test датасета в resources/summarization_ds_test.parquet...")
    test_df.to_parquet("resources/summarization_ds_test.parquet", index=False)

    print("Датасет успешно сохранён:")
    print("→ resources/summarization_ds_train.parquet")
    print("→ resources/summarization_ds_test.parquet")

if __name__ == "__main__":
    save_dataset_as_parquet()
