#!/bin/bash
set -euo pipefail

if [ -e "./merged_model" ]; then
    sudo rm -rf "./merged_model"
fi
if [ -e "./gemma_tuned_remote_75k.gguf" ]; then
    sudo rm -rf "./gemma_tuned_remote_75k.gguf"
fi


echo "Запуск мерджа модели"
python merge_model.py

echo "Запуск конвертации в GGUF"
python convert_to_gguf.py

MERGED_GGUF="gemma_tuned_remote_75k.gguf"
OLLAMA_BLOBS_DIR="../../ollama-docker/ollama/ollama/models/gemma_75_remote"

echo "Копирование GGUF модели в Ollama models"
sudo mkdir -p "$OLLAMA_BLOBS_DIR"
if [ -e "$OLLAMA_BLOBS_DIR/$MERGED_GGUF" ]; then
    sudo rm "$OLLAMA_BLOBS_DIR/$MERGED_GGUF"
fi
sudo cp "$MERGED_GGUF" "$OLLAMA_BLOBS_DIR/"

# Название контейнера Ollama (замени при необходимости)
OLLAMA_CONTAINER_NAME="ollama"

echo "Удаление старой модели в Ollama (если есть)"
docker exec "$OLLAMA_CONTAINER_NAME" ollama rm gemma3-1b-ftr-75k || true

echo "Создание новой модели в Ollama"
docker exec "$OLLAMA_CONTAINER_NAME" ollama create gemma3-1b-ftr-75k -f /root/.ollama/models/gemma_75_remote/Modelfile

echo "Готово"
