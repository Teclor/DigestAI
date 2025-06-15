import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime

import pandas as pd
import requests
from rouge import Rouge
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoTokenizer

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from config import CONFIG

os.makedirs(CONFIG["output"]["results_dir"], exist_ok=True)

LOCAL_DATASET_PATH = os.path.join(CONFIG["output"]["results_dir"], "dataset_cache.parquet")
SAVE_INTERVAL = CONFIG.get("save_interval", 10)

# Файлы вывода
METRICS_FILE = CONFIG["output"]["metrics_file"]
BACKUP_METRICS_FILE = CONFIG["output"]["backup_metrics_file"]
FAILED_SUMMARIES_FILE = CONFIG["output"]["failed_summaries_file"]
SUMMARIES_JSON = os.path.join(CONFIG["output"]["results_dir"], "summaries.json")

ROUGE_METRIC = Rouge()
MAX_TOKENS = 768

def load_dataset():
    if os.path.exists(LOCAL_DATASET_PATH):
        logger.info(f"Используем локальный датасет: {LOCAL_DATASET_PATH}")
        df = pd.read_parquet(LOCAL_DATASET_PATH)
    else:
        logger.info("Загружаем датасет с HuggingFace")
        splits = {'train': 'train/train.parquet', 'test': 'test/test.parquet'}
        df = pd.read_parquet(f"hf://datasets/{CONFIG['dataset']['name']}/{splits[CONFIG['dataset']['split']]}")
        df = df[["text", "summary"]].dropna()

        df.to_parquet(LOCAL_DATASET_PATH)
        logger.info(f"Датасет сохранён локально: {LOCAL_DATASET_PATH}")

    tokenizer = AutoTokenizer.from_pretrained("unsloth/gemma-3-1b-it-qat", trust_remote_code=True)

    def count_tokens(text, summary):
        try:
            chat_full = [
                {"role": "user", "content": f"Перескажи следующий текст:\n{text}"},
                {"role": "assistant", "content": summary}
            ]
            tokenized = tokenizer.apply_chat_template(
                chat_full,
                add_generation_prompt=False,
                return_tensors="pt"
            )
            return tokenized.shape[1]  # Возвращаем длину последовательности
        except Exception as e:
            logger.warning(f"Ошибка при токенизации примера: {e}")
            return MAX_TOKENS + 1  # Исключаем сомнительные случаи

    df = df.iloc[-500:].reset_index(drop=True)

    logger.info("Фильтруем датасет по длине токенов...")
    indices_to_keep = []
    for idx, row in df.iterrows():
        length = count_tokens(row["text"], row["summary"])
        if length <= MAX_TOKENS:
            indices_to_keep.append(idx)

    filtered_df = df.iloc[indices_to_keep].reset_index(drop=True)

    logger.info(f"Оригинальный размер датасета: {len(df)} записей")
    logger.info(f"После фильтрации: {len(filtered_df)} записей (≤ {MAX_TOKENS} токенов)")

    return filtered_df


def preprocess_text(text):
    # Очистка текста от шума
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'[^а-яА-Яa-zA-Z0-9\s.,!?:;]', '', text)
    return text


def get_model_summary(model_name, text):
    url = CONFIG["ollama_api_url"]

    payload = {
        "model": model_name,
        "prompt": text,
        "stream": False,
# Старый промпт. Использовался для отбора моделей, когда не было лимита по токенам
#         "system": f"""
# Ты — эксперт по пересказу текстов. Твоя задача:
# 1. Игнорировать служебные строки в начале текста (например: "Кратко суммаризируй", "Какова основная идея" и т.п.).
# 2. Создать предельно краткий пересказ, который:
#    - Сохраняет полную смысловую нагрузку оригинала (что происходит, кто участвует, зачем, как, к чему это приводит).
#    - Исключает любые детали, не влияющие на суть (даты, названия, повторяющиеся слова).
#    - Использует минимальное количество слов, но не менее 5 слов (даже если текст очень короткий). Но можешь сохранить ключевые термины из оригинала, если это помогает точности пересказа.
#    - Избегает механического повторения целых предложений.
#    - Структурирует информацию:
#      - Если текст описывает процесс — сохрани логическую цепочку действий.
#      - Если текст описывает проблему — выдели её суть и возможные решения.
# 3. Проверь пересказ:
#    - Если в тексте есть прямая речь, то замени её на пересказ с помощью косвенной речи. Пример: "Олег: Привет! Как дела?" изменится на "Олег поприветствовал и спросил как дела"
#    - Если, можно убрать слово без потери смысла, то убирай.
#    - Если можно заменить слово на более ёмкий термин, то заменяй.
#    - Если есть повторы, то упрощай и избавляйся от повторов.
# 4. Ответ должен содержать только пересказ, без объяснений, форматирования, маркированных списков. В ответе не должно быть вводных фраз по типу "объясняем о чем говорится", "кратко пересказываем" и им подобных.
# 5. Пересказ должен быть на русском языке.
# """.strip()
        "system": "Перескажи текст кратко, сохраняя его смысл. В ответе должен быть только сам пересказ."
    }

    pause_on_error=True
    for attempt in range(2):
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=60)
            elapsed = time.time() - start_time

            if response.status_code == 200:
                result = response.json()
                summary = result.get("response", "").strip()
                if summary:
                    return summary, elapsed
                logger.warning(f"Модель {model_name} вернула пустой ответ. Повтор...")
            else:
                logger.error(f"Ошибка у модели {model_name}: {response.status_code}")
                if response.status_code >= 500:
                    logger.info("Сервис недоступен. Ожидание... Нажмите Enter для повторной попытки.")
                    if pause_on_error:
                        input()
                continue

        except requests.exceptions.Timeout as e:
            logger.warning(f"Таймаут у модели {model_name}.")
            logger.info("Нажмите Enter для повторной попытки")
            if pause_on_error:
                input()
            continue

        except requests.exceptions.ConnectionError as e:
            logger.error(f"Нет подключения к сервису {model_name}. Проверьте ollama и сеть")
            logger.info("Ожидание... Нажмите Enter для продолжения")
            if pause_on_error:
                input()
            continue

        except Exception as e:
            logger.error(f"Неизвестная ошибка у модели {model_name}: {e}")
            logger.info("Нажмите Enter для повторной попытки")
            if pause_on_error:
                input()
            continue

    # Если все попытки неудачны — сохраняем провал
    save_failed_example(model_name, text)
    return "", float('inf')


def calculate_metrics(original, reference, summary, elapsed):
    if not isinstance(original, str) or not isinstance(summary, str):
        return {
            "rouge_1": 0,
            "rouge_2": 0,
            "rouge_l": 0,
            "jaccard": 0,
            "cosine": 0,
            "compression": 0,
            "avg_speed_per_token": 0,
            "time": round(elapsed, 2)
        }

    if not original.strip() or not summary.strip():
        return {
            "rouge_1": 0,
            "rouge_2": 0,
            "rouge_l": 0,
            "jaccard": 0,
            "cosine": 0,
            "compression": 0,
            "avg_speed_per_token": 0,
            "time": round(elapsed, 2)
        }

    # ROUGE: между пересказом модели и эталоном
    try:
        rouge_scores = ROUGE_METRIC.get_scores(summary, reference)[0]
    except Exception as e:
        logger.warning(f"Ошибка при подсчёте ROUGE: {e}")
        rouge_scores = {"rouge-1": {"f": 0}, "rouge-2": {"f": 0}, "rouge-l": {"f": 0}}

    # Compression Ratio: между пересказом и оригиналом
    orig_tokens = len(original.split())
    summ_tokens = len(summary.split())
    comp_ratio = summ_tokens / orig_tokens if orig_tokens > 0 else 1

    # Speed per Token: между временем и оригиналом
    speed_per_token = elapsed / orig_tokens if orig_tokens > 0 else 0

    # Cosine Similarity: между пересказом модели и эталоном
    try:
        tfidf_matrix = TfidfVectorizer().fit_transform([reference, summary])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
    except:
        cosine_sim = 0

    # Jaccard Similarity: между пересказом модели и эталоном
    def jaccard(s1, s2):
        set1, set2 = set(s1.split()), set(s2.split())
        return len(set1 & set2) / len(set1 | set2) if set1 or set2 else 0

    return {
        "rouge_1": rouge_scores["rouge-1"]["f"],
        "rouge_2": rouge_scores["rouge-2"]["f"],
        "rouge_l": rouge_scores["rouge-l"]["f"],
        "jaccard": jaccard(reference, summary),
        "cosine": cosine_sim,
        "compression": comp_ratio,  # ← Теперь на основе оригинального текста
        "avg_speed_per_token": speed_per_token,  # ← На основе оригинального текста
        "time": round(elapsed, 2)
    }

def benchmark_models(models, dataset, max_runtime):
    results = {model: {"metrics": [], "times": []} for model in models}
    start_time = datetime.now()
    total_texts = 0

    logger.info(f"Начало тестирования в {start_time.strftime('%H:%M:%S')}")
    logger.info(f"Максимальное время тестирования: {max_runtime.total_seconds() // 60} мин")

    last_state = load_last_processed_index(dataset)
    start_idx = last_state["last_index"] + 1 if last_state else 0
    logger.info(f"Стартуем с текста #{start_idx}")
    current_summaries = {}

    for idx, row in enumerate(dataset.itertuples(), start_idx):
        current_time = datetime.now()
        elapsed_minutes = (current_time - start_time).total_seconds() / 60
        remaining_minutes = (max_runtime - (current_time - start_time)).total_seconds() / 60

        if remaining_minutes < 0:
            logger.info("Истекло время тестирования")
            break

        logger.info(f"[+] Прошло: {elapsed_minutes:.1f} мин | Осталось: {remaining_minutes:.1f} мин")

        try:
            original = preprocess_text(row.text)
            reference = preprocess_text(row.summary)
        except Exception as e:
            logger.warning(f"Ошибка при обработке текста #{idx}: {e}")
            continue

        # Генерируем уникальный ID для текста
        text_unique_id = hashlib.md5(original.encode()).hexdigest()

        logger.info(f"\nОбработка текста #{idx}")
        total_texts += 1

        current_summaries = {
            "text_id": text_unique_id,
            "original": original,
            "reference": reference,
            "models": {}
        }

        for model in models:
            if (datetime.now() - start_time).seconds > max_runtime.total_seconds():
                break

            logger.info(f"Модель: {model}")
            summary, elapsed = get_model_summary(model, original)

            if not summary.strip():
                logger.warning(f"Модель {model} вернула пустой пересказ. Пропускаем...")
                continue

            summary = remove_think_tag(summary)

            # Вычисляем метрики
            metrics = calculate_metrics(original, reference, summary, elapsed)
            metrics["text_id"] = text_unique_id
            metrics["model"] = model
            metrics["time"] = elapsed
            results[model]["metrics"].append(metrics)
            results[model]["times"].append(elapsed)

            # Сохраняем пересказ модели
            current_summaries["models"][model] = {
                "generated": summary,
                "time": elapsed
            }

            logger.info(f"Время: {elapsed:.2f} сек. Метрики: {metrics}")

        # Сохранение каждые N итераций
        if idx % SAVE_INTERVAL == 0 and idx != 0:
            aggregated = aggregate_results(results)
            save_metrics(aggregated, append=True)
            save_backup_metrics(aggregated)
            save_last_summaries(current_summaries)
            save_last_processed_index(idx)
            logger.info(f"[+] Сохранено после {idx} текстов")

    # Агрегация результатов
    logger.info("Агрегация результатов")
    aggregated = aggregate_results(results)
    save_metrics(aggregated, append=True)
    save_backup_metrics(aggregated)
    save_last_summaries(current_summaries)
    save_last_processed_index(len(dataset))
    logger.info("✅ Тестирование завершено успешно!")

    return aggregated


def aggregate_results(results):
    aggregated = {}

    for model, data in results.items():
        if not data["metrics"]:
            continue

        df = pd.DataFrame(data["metrics"])
        avg_metrics = {
            "time": df["time"].mean(),
            "rouge_1": df.get("rouge_1", pd.Series([0])).mean(),
            "rouge_2": df.get("rouge_2", pd.Series([0])).mean(),
            "rouge_l": df.get("rouge_l", pd.Series([0])).mean(),
            "jaccard": df.get("jaccard", pd.Series([0])).mean(),
            "cosine": df.get("cosine", pd.Series([0])).mean(),
            "compression": df.get("compression", pd.Series([0])).mean(),
            "avg_speed_per_token": df.get("avg_speed_per_token", pd.Series([0])).mean()
        }

        processed_count = len(data["metrics"])

        last_text_id = data["metrics"][-1].get("text_id", None)

        aggregated[model] = {
            "total_processed": processed_count,
            "last_text_id": last_text_id,
            "avg_time": round(avg_metrics["time"], 2),
            "rouge_1": round(avg_metrics["rouge_1"], 4),
            "rouge_2": round(avg_metrics["rouge_2"], 4),
            "rouge_l": round(avg_metrics["rouge_l"], 4),
            "jaccard": round(avg_metrics["jaccard"], 4),
            "cosine": round(avg_metrics["cosine"], 4),
            "compression": round(avg_metrics["compression"], 4),
            "avg_speed_per_token": round(avg_metrics["avg_speed_per_token"], 6),
            "history": data["metrics"]  # ✅ Сохраняем все метрики как история
        }

    return aggregated


def save_metrics(metrics, append=True):
    existing = {}

    if append and os.path.exists(METRICS_FILE):
        try:
            with open(METRICS_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            logger.warning("Файл metrics.json повреждён или отсутствует. Создаётся новый.")

    # Для каждой модели проверяем, есть ли уже этот text_id в истории
    for model_name, model_data in metrics.items():
        history_to_add = model_data.get("history", [])

        if model_name not in existing:
            existing[model_name] = {
                "total_processed": len(history_to_add),
                "last_update": datetime.now().isoformat(),
                "history": history_to_add
            }
        else:
            existing_ids = {entry["text_id"] for entry in existing[model_name].get("history", [])}
            new_history = [entry for entry in history_to_add if entry.get("text_id") not in existing_ids]

            # Обновляем историю и количество
            existing[model_name]["history"] = existing[model_name].get("history", []) + new_history
            existing[model_name]["total_processed"] = len(existing[model_name]["history"])
            existing[model_name]["last_update"] = datetime.now().isoformat()

    # Сохраняем обратно в файл
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(existing, f, ensure_ascii=False, indent=2)

    logger.info(f"Метрики сохранены в {METRICS_FILE}")


def save_backup_metrics(metrics):
    with open(BACKUP_METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info(f"Резервная копия метрик сохранена в {BACKUP_METRICS_FILE}")

def save_last_summaries(summaries_entry):
    file_exists = os.path.exists(SUMMARIES_JSON)

    with open(SUMMARIES_JSON, "a", encoding="utf-8") as f:
        if not file_exists or os.path.getsize(SUMMARIES_JSON) == 0:
            # Если файл пустой, начинаем массив
            f.write("[\n")
            json.dump(summaries_entry, f, ensure_ascii=False, indent=2)
            f.write("\n]")
        else:
            # Убираем ] в конце, добавляем запятую и новый объект
            f.seek(0, os.SEEK_END)
            f.seek(f.tell() - 2)  # уходим перед '\n]'
            f.truncate()

            f.write(",\n")
            json.dump(summaries_entry, f, ensure_ascii=False, indent=2)
            f.write("\n]")

    logger.info(f"Пересказ сохранён в {SUMMARIES_JSON}")


def save_failed_example(model_name, original_text):
    # Сохраняет провальные примеры
    failed = {"model": model_name, "original": original_text}
    with open(FAILED_SUMMARIES_FILE, "a", encoding="utf-8") as f:
        json.dump(failed, f, ensure_ascii=False)
        f.write("\n")


def load_last_processed_index(dataset):
    if not os.path.exists(METRICS_FILE):
        return None

    try:
        with open(METRICS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
            if data and data.values():
                processed = next(iter(data.values()))["processed_texts"]
                return {"last_index": processed}
    except json.JSONDecodeError as e:
        logger.warning(f"Файл {METRICS_FILE} повреждён или пустой: {e}")
    except Exception as e:
        logger.warning(f"Не удалось загрузить прогресс: {e}")

    return None


def save_last_processed_index(index):
    progress_path = os.path.join(CONFIG["output"]["results_dir"], "progress.json")
    with open(progress_path, "w", encoding="utf-8") as f:
        json.dump({"last_index": index}, f)


def remove_think_tag(summary):
    # Удаляем think-блоки из ответа модели
    return re.sub(r"<think>.*?</think>\n+", "", summary, flags=re.DOTALL)


def main():
    try:
        logger.info("Загрузка датасета...")
        df = load_dataset()
        logger.info(f"Загружено {len(df)} текстов")

        logger.info("Начало тестирования моделей")
        benchmark_models(CONFIG["models"], df, CONFIG["max_runtime"])

    except Exception as e:
        logger.error(f"Ошибка выполнения: {str(e)}")


if __name__ == "__main__":
    main()