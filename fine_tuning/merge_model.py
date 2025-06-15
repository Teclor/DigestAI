import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL = "unsloth/gemma-3-1b-it-qat"
PEFT_MODEL_PATH = "output_remote"
MERGED_MODEL_PATH = "merged_model"

def merge_lora_and_save():
    print("Загружаем базовую модель...")
    base_model = AutoModelForCausalLM.from_pretrained(BASE_MODEL, torch_dtype="auto")

    print("Загружаем PEFT (LoRA) веса...")
    peft_model = PeftModel.from_pretrained(base_model, PEFT_MODEL_PATH)

    print("Сливаем LoRA в базовую модель...")
    merged_model = peft_model.merge_and_unload()

    print(f"Сохраняем объединённую модель в: {MERGED_MODEL_PATH}")
    os.makedirs(MERGED_MODEL_PATH, exist_ok=True)
    merged_model.save_pretrained(MERGED_MODEL_PATH)

    print("Сохраняем токенизатор...")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
    tokenizer.save_pretrained(MERGED_MODEL_PATH)

    print("Объединение и сохранение модели успешно завершено!")

if __name__ == "__main__":
    merge_lora_and_save()
