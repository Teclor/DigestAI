import os

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import torch
import pandas as pd
from datasets import Dataset, load_from_disk
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq,
    BitsAndBytesConfig,
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

torch.cuda.empty_cache()

MODEL_NAME = "unsloth/gemma-3-1b-it-qat"
OUTPUT_DIR = "output"
# MAX_SEQ_LENGTH - Общая длина в токенах. Промпт, текст, пересказ
MAX_SEQ_LENGTH = 768
TOKENIZED_DIR = "resources/tokenized_data_summarization"

try:
    train_df = pd.read_parquet("resources/summarization_ds_train.parquet")
    test_df = pd.read_parquet("resources/summarization_ds_test.parquet")

    train_dataset = Dataset.from_pandas(train_df)
    test_dataset = Dataset.from_pandas(test_df)
except Exception as e:
    print(f"Ошибка при загрузке датасета: {e}")
    raise

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, trust_remote_code=True)

# Устанавливаем pad_token для токенизатора, если он не задан
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
        print(f"Установлен pad_token = eos_token: {tokenizer.eos_token}")
    else:
        # Если нет eos_token, добавляем новый pad_token
        tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        print("Добавлен новый pad_token: [PAD]")

def tokenize_function(examples):
    input_ids = []
    attention_masks = []
    labels = []

    for i in range(len(examples["text"])):
        try:
            # В датасете уже содержится промпт для пересказа текста
            chat_user_content = examples['text'][i]

            # Применяем шаблон чата для формирования входа модели
            chat_full = [
                {"role": "user", "content": chat_user_content},
                {"role": "assistant", "content": examples['summary'][i]}
            ]

            tokenized_full = tokenizer.apply_chat_template(
                chat_full,
                add_generation_prompt=False,  # Мы хотим получить весь диалог
                padding="max_length",
                truncation=True,
                max_length=MAX_SEQ_LENGTH,
                return_tensors="pt"
            ).to("cpu")

            # Получаем длину prompt части (до ответа модели)
            chat_prefix = [{"role": "user", "content": chat_user_content}]
            tokenized_prefix = tokenizer.apply_chat_template(
                chat_prefix,
                add_generation_prompt=True,
                padding=False,
                truncation=True,
                return_tensors="pt"
            ).to("cpu")

            prefix_length = tokenized_prefix.shape[-1]

            # Маскируем часть текста (prompt) меткой -100, чтобы модель не училась на ней.
            # -100 — специальное значение в PyTorch, игнорируемое при подсчёте loss.
            label_ids = tokenized_full.clone()
            label_ids[:, :prefix_length] = -100

            # Если есть padding — тоже маскируем
            pad_token_id = tokenizer.pad_token_id
            if pad_token_id is not None:
                pad_indices = (tokenized_full == pad_token_id).nonzero(as_tuple=True)
                if len(pad_indices) > 1:
                    label_ids[pad_indices[0], pad_indices[1]] = -100

            input_ids.append(tokenized_full.squeeze().tolist())
            attention_masks.append((tokenized_full != pad_token_id).long().squeeze().tolist())
            labels.append(label_ids.squeeze().tolist())

        except Exception as e:
            print(f"Ошибка при обработке примера {i}: {e}")
            # В случае ошибки добавляем пустые/недействительные данные
            input_ids.append([tokenizer.pad_token_id] * MAX_SEQ_LENGTH)
            attention_masks.append([0] * MAX_SEQ_LENGTH)
            labels.append([-100] * MAX_SEQ_LENGTH)

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels
    }


if os.path.exists(TOKENIZED_DIR) and os.path.exists(os.path.join(TOKENIZED_DIR, "train")) and os.path.exists(
        os.path.join(TOKENIZED_DIR, "test")):
    print("Загрузка токенизированного датасета из кэша...")
    try:
        tokenized_train = load_from_disk(os.path.join(TOKENIZED_DIR, "train"))
        tokenized_test = load_from_disk(os.path.join(TOKENIZED_DIR, "test"))
        print("Токенизированный датасет успешно загружен из кэша.")
    except Exception as e:
        print(f"Ошибка загрузки кэшированного датасета: {e}. Токенизация с нуля...")
        # Удаляем некорректную папку кэша
        import shutil

        if os.path.exists(TOKENIZED_DIR):
            shutil.rmtree(TOKENIZED_DIR)
        os.makedirs(TOKENIZED_DIR, exist_ok=True)

        print("Токенизация обучающего датасета...")
        tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
        print("Токенизация тестового датасета...")
        tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

        print("Сохранение токенизированного датасета...")
        tokenized_train.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
        tokenized_test.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))
        print("Токенизированный датасет сохранен.")
else:
    print("Токенизация с нуля...")
    os.makedirs(TOKENIZED_DIR, exist_ok=True)  # Создаем папку, если ее нет

    print("Токенизация обучающего датасета...")
    tokenized_train = train_dataset.map(tokenize_function, batched=True, remove_columns=train_dataset.column_names)
    print("Токенизация тестового датасета...")
    tokenized_test = test_dataset.map(tokenize_function, batched=True, remove_columns=test_dataset.column_names)

    print("Сохранение токенизированного датасета...")
    tokenized_train.save_to_disk(os.path.join(TOKENIZED_DIR, "train"))
    tokenized_test.save_to_disk(os.path.join(TOKENIZED_DIR, "test"))
    print("Токенизированный датасет сохранен.")


# Загрузка модели с квантованием и PEFT
compute_dtype = torch.float16

bnb_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=compute_dtype,
    bnb_4bit_use_double_quant=True,
    bnb_4bit_quant_type="nf4"  # nf4 обычно лучше для точности
)

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    quantization_config=bnb_config,
    trust_remote_code=True,
    attn_implementation="eager",  # По совету от разработчиков gemma3. Но "flash_attention_2" лучше в плане оптимизации.
    device_map="auto",
    torch_dtype=compute_dtype
)

model.config.use_cache = False
model.gradient_checkpointing_enable()

model.to("cuda")  # Переносим на GPU

model = prepare_model_for_kbit_training(
    model,
    gradient_checkpointing_kwargs={"use_reentrant": False}
)

peft_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=[
        "q_proj", "k_proj", "v_proj", "o_proj",  # Attention layers
        "gate_proj", "up_proj", "down_proj"  # MLP layers
    ]
)

model = get_peft_model(model, peft_config)
model.print_trainable_parameters()  # Вывод, какая часть модели обучается

training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=1, # для обученной модели достаточно одной эпохи
    per_device_train_batch_size=16,  # Оптимальный batch-size 16, вычисляется как gradient_accumulation_steps * per_device_train_batch_size
    per_device_eval_batch_size=16,
    gradient_accumulation_steps=1,
    optim="adamw_torch_fused" if torch.cuda.is_available() and hasattr(torch.optim,
                                                                       'AdamW') and 'fused' in torch.optim.AdamW.__init__.__kwdefaults__ else "adamw_torch",
    learning_rate=2e-4,  # Типичное значение для QLoRA.
    lr_scheduler_type="cosine",
    warmup_ratio=0.05,
    # 5% шагов на разогрев.
    # warmup_steps = int(total_train_steps * warmup_ratio)
    # warmup_steps = int((75000 / (16*1)) * 1 * 0.05) ~ 250 шагов
    warmup_steps=250,

    fp16=(compute_dtype == torch.float16),
    bf16=(compute_dtype == torch.bfloat16),

    logging_strategy="steps",
    logging_steps=100,
    eval_strategy="steps",
    eval_steps=500,
    save_strategy="steps",
    save_steps=500,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,

    report_to=["tensorboard"],
    dataloader_num_workers=4,
    label_names=["labels"],
    remove_unused_columns=True,  # Удалять неиспользуемые колонки из датасета
    max_grad_norm=0.3, # Клиппинг градиента для стабильности
    weight_decay=0.01, # Небольшая регуляризация L2
    max_steps=-1,  # Если > 0, переопределяет num_train_epochs
)

# label_pad_token_id=-100 говорит data collator заменять padding в labels на -100
data_collator = DataCollatorForSeq2Seq(
    tokenizer,
    model=model,
    label_pad_token_id=-100,  # Игнорировать pad_token в вычислении лосса
    pad_to_multiple_of=8  # Возможно небольшое ускорение на некоторых GPU
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_test,
    data_collator=data_collator,
    tokenizer=tokenizer,
)

# Запуск обучения
checkpoint_path = None
if os.path.isdir(OUTPUT_DIR):
    checkpoint_files = [
        os.path.join(OUTPUT_DIR, d)
        for d in os.listdir(OUTPUT_DIR)
        if d.startswith("checkpoint-") and os.path.isdir(os.path.join(OUTPUT_DIR, d))
    ]
    if checkpoint_files:
        checkpoint_files.sort(key=lambda x: int(x.split("-")[-1]))
        checkpoint_path = checkpoint_files[-1]
        print(f"Возобновление обучения с чекпоинта: {checkpoint_path}")

print("Начало обучения...")
try:
    trainer.train(resume_from_checkpoint=checkpoint_path)
except Exception as e:
    print(f"Произошла ошибка во время обучения: {e}")
    raise

# === Сохранение модели и адаптеров ===
print(f"Сохранение лучшей модели в {OUTPUT_DIR}")
trainer.save_model(OUTPUT_DIR)  # Сохраняет лучшую модель (адаптеры LoRA)
tokenizer.save_pretrained(OUTPUT_DIR)  # Сохраняет токенизатор

print("Обучение завершено!")