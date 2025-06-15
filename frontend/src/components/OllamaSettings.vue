<template>
  <div class="ollama-settings bg-gray-800 rounded-xl shadow-md p-4 mb-6">
    <!-- Заголовок и кнопка открытия -->
    <div class="flex justify-between items-center cursor-pointer" @click="toggle">
      <h3 class="text-lg font-semibold text-white">Настройки Ollama</h3>
      <button class="text-green-400 hover:text-green-300 transition">
        {{ isOpen ? '▼' : '▶' }}
      </button>
    </div>

    <!-- Контент спойлера -->
    <div v-show="isOpen" class="mt-4 space-y-4">
      <!-- Поле ввода для URL -->
      <div class="form-group">
        <label for="ollama-url" class="block text-sm font-medium text-gray-300 mb-1">Ollama Host:</label>
        <input
          type="text"
          id="ollama-url"
          v-model="ollamaUrl"
          placeholder="http://localhost:11434/api/generate"
          class="w-full px-4 py-2 bg-gray-700 text-white border-none rounded-lg focus:ring-2 focus:ring-green-500"
        />
      </div>

      <!-- Выбор модели -->
      <div class="form-group">
        <label for="ollama-model" class="block text-sm font-medium text-gray-300 mb-1">Выберите модель:</label>
        <select
          id="ollama-model"
          v-model="selectedModel"
          class="w-full px-4 py-2 bg-gray-700 text-white border-none rounded-lg focus:ring-2 focus:ring-green-500"
        >
          <option v-for="model in models" :key="model" :value="model">{{ model }}</option>
        </select>
      </div>

      <!-- Единая кнопка сохранения -->
      <div class="form-actions">
        <button
          @click="saveSettings"
          class="w-full px-4 py-2 bg-gradient-to-r from-green-600 to-emerald-500 text-white rounded-lg hover:from-green-700 hover:to-emerald-600 transition"
        >
          Сохранить настройки
        </button>
      </div>
    </div>
  </div>
</template>

<script>
import api from '@/services/api'

export default {
  data() {
    return {
      isOpen: false,
      ollamaUrl: '',
      selectedModel: '',
      models: []
    }
  },
  async mounted() {
    await this.loadCurrentSettings()
  },
  methods: {
    toggle() {
      this.isOpen = !this.isOpen
    },
    async loadCurrentSettings() {
      try {
        const [url, models] = await Promise.all([
          api.getOllamaUrl(),
          api.getOllamaModels()
        ])

        this.ollamaUrl = url
        this.models = models.models
        this.selectedModel = models.current_model
      } catch (err) {
        console.error('Ошибка загрузки настроек:', err)
      }
    },
    async saveSettings() {
      if (!this.ollamaUrl) {
        alert('Введите корректный URL')
        return
      }

      if (!this.models.includes(this.selectedModel)) {
        alert('Выберите корректную модель')
        return
      }

      try {
        // Отправляем оба значения за один раз
        await Promise.all([
          api.setOllamaUrl(this.ollamaUrl),
          api.setOllamaModel(this.selectedModel)
        ])

        console.log('Настройки успешно сохранены')
      } catch (err) {
        alert('Ошибка при сохранении настроек: ' + (err.response?.data?.detail || err.message))
      }
    }
  }
}
</script>

<style scoped>
.ollama-settings {
  background-color: #1a1a1a;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.form-group {
  margin-bottom: 1.5rem;
}

.input-field {
  width: 100%;
  padding: 0.75rem 1rem;
  font-size: 1rem;
  background-color: #2a2a2a;
  color: #e6e6e6;
  border: none;
  border-radius: 8px;
  outline: none;
  transition: background-color 0.3s ease;
}

.input-field:focus {
  background-color: #333;
}

.form-actions {
  margin-top: 1rem;
}

.btn-save {
  display: inline-block;
  width: 100%;
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  color: white;
  background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
}

.btn-save:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
}
</style>