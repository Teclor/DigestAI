#!/bin/bash

set -e  # Остановить при ошибке

echo "Обновление системы и установка зависимостей..."
sudo apt update && sudo apt upgrade -y
sudo apt install -y git curl wget build-essential python3 python3-pip python3-venv libgl1 libsm6 ffmpeg unzip zip

if command -v lspci &> /dev/null; then
    gpu_check=$(lspci | grep -i nvidia)
    if [[ $gpu_check ]]; then
        echo "🎮 Найдена NVIDIA GPU. Установка драйверов..."
        sudo apt install -y nvidia-driver-535 nvidia-cuda-toolkit
        echo "✅ Драйверы NVIDIA установлены."
        nvidia-smi || echo "nvidia-smi не запущен — возможно, требуется перезагрузка"
    else
        echo "❌ GPU не обнаружена. Продолжаю без установки драйверов NVIDIA."
    fi
else
    echo "'lspci' не найден. Пропуск проверки GPU."
fi

echo "Создание виртуального окружения..."
cd ~
python3 -m venv ~/venv_gemma
source ~/venv_gemma/bin/activate

echo "Обновление pip..."
pip install --upgrade pip

echo "Установка PyTorch и дополнительных модулей..."
pip install -r ./requirements.txt
echo "Создание рабочей директории..."
mkdir -p ~/gemma_project
cp ~/train.py ~/gemma_project/ 2>/dev/null || echo "train.py не найден — загрузите его вручную"

echo "Проверка GPU..."
python3 -c "
import torch;
print('CUDA доступна:', torch.cuda.is_available());
print('Количество GPU:', torch.cuda.device_count());
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        print(f'Device {i}:', torch.cuda.get_device_name(i))
"

echo "Запуск обучения..."
cd ~/gemma_project

export CUDA_VISIBLE_DEVICES=0

# Активируем venv перед запуском
source ~/venv_gemma/bin/activate

nohup python3 -u train.py > training.log 2>&1 &
echo "PID процесса обучения: $!"
echo "Логи сохраняются в training.log"
