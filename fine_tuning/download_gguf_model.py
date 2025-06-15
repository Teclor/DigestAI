import os
from huggingface_hub import hf_hub_download

# Настройки
repo_id = "unsloth/gemma-3-1b-it-qat"
filename = "gemma-1b-it-qat.gguf"
local_dir = os.path.expanduser("./gemma3-1b-it-qat")
os.makedirs(local_dir, exist_ok=True)

print(f"[+] Скачивание модели {repo_id}/{filename}...")

try:
    downloaded_path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        local_dir=local_dir,
        local_files_only=False,
        revision="main"
    )

    print(f"[+] Модель успешно сохранена: {downloaded_path}")

except Exception as e:
    print(f"[!] Ошибка при скачивании модели: {e}")