# graph.py
import os
import json
from collections import defaultdict

import matplotlib.pyplot as plt
import pandas as pd
import logging
from config import CONFIG
from datetime import datetime
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

METRICS_JSON_PATH = CONFIG["output"]["metrics_file"]
RESULTS_DIR = CONFIG["output"]["results_dir"]


def load_metrics_history():
    """Загружает всю историю метрик из файла"""
    if not os.path.exists(METRICS_JSON_PATH):
        logger.error(f"Файл метрик не найден: {METRICS_JSON_PATH}")
        exit(1)

    with open(METRICS_JSON_PATH, "r", encoding="utf-8") as f:
        try:
            data = json.load(f)
        except json.JSONDecodeError:
            logger.error("Файл метрик повреждён или пустой")
            exit(1)

    # Для каждой модели собираем усреднённые данные
    latest_data = {}

    for model_name, model_data in data.items():
        history = model_data.get("history", [])
        if not history:
            continue

        df = pd.DataFrame(history)

        # Усредняем по всем записям для модели
        avg_time = df["time"][~np.isinf(df["time"])].mean()
        rouge_1 = df["rouge_1"].mean()
        rouge_2 = df["rouge_2"].mean()
        rouge_l = df["rouge_l"].mean()
        jaccard = df["jaccard"].mean()
        cosine = df["cosine"].mean()
        compression = df["compression"].mean()
        avg_speed_per_token = df["avg_speed_per_token"].mean()

        latest_data[model_name] = {
            "processed_texts": model_data.get("total_processed", 0),
            "avg_time": round(avg_time, 2),
            "rouge_1": round(rouge_1, 4),
            "rouge_2": round(rouge_2, 4),
            "rouge_l": round(rouge_l, 4),
            "jaccard": round(jaccard, 4),
            "cosine": round(cosine, 4),
            "compression": round(compression, 4),
            "avg_speed_per_token": round(avg_speed_per_token, 4)
        }

    return latest_data


def plot_comparison(data):
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))

    # Подсчитываем общее число уникальных текстов
    total_texts = max(model_data.get("processed_texts", 0) for model_data in data.values()) if data else 0

    logger.info(f"Всего обработано текстов: {total_texts}")

    metrics_to_plot = ['rouge_1', 'rouge_2', 'rouge_l', 'jaccard', 'cosine', 'avg_time', 'avg_speed_per_token', 'compression']
    titles = ['ROUGE-1', 'ROUGE-2', 'ROUGE-L', 'Jaccard', 'Cosine', 'Среднее время', 'Время/токен', 'Сжатие']

    values = defaultdict(list)
    for metric in metrics_to_plot:
        for model in data:
            entry = data[model]
            values[metric].append({
                "model": model,
                metric: entry.get(metric, 0)
            })

    for ax, metric, title in zip(axes.flat, metrics_to_plot, titles):
        df = pd.DataFrame(values[metric])
        df.set_index("model", inplace=True)
        df.sort_values(by=metric, ascending=False).plot(kind='bar', y=metric, ax=ax, title=title, legend=False)
        ax.set_ylabel("")
        ax.tick_params(axis='x', rotation=45)

    plt.tight_layout()
    PLOT_OUTPUT_PATH = os.path.join(RESULTS_DIR, f"comparison_n{total_texts}_{datetime.now().strftime('%d_%H%M')}.png")
    plt.savefig(PLOT_OUTPUT_PATH, bbox_inches='tight')
    logger.info(f"График сохранён в {PLOT_OUTPUT_PATH}")

def main():
    logger.info("Загрузка истории метрик...")
    metrics_history = load_metrics_history()

    logger.info("Построение графика...")
    plot_comparison(metrics_history)


if __name__ == "__main__":
    main()