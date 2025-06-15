<template>
  <div v-if="messages.length > 0" class="summary-block">
    <div class="text-block">
      <h3>Текст для пересказа</h3>
      <textarea
        v-model="truncatedText"
        class="truncated-input"
        placeholder="Подготовленный текст будет здесь..."
      ></textarea>

      <button class="btn-clear" @click="clearSummary">Очистить</button>
      <button class="btn-summarize" @click="getSummary">Пересказать</button>
    </div>

    <div v-if="summary" class="summary-result">
      <h3>Результат пересказа</h3>
      <pre class="summary-text">{{ summary }}</pre>
    </div>
  </div>
</template>

<script>
import api from '@/services/api'

export default {
  props: {
    messages: {
      type: Array,
      required: true
    },
    selectedTopic: {
      type: String,
      default: 'chat_1'
    },
    // Новый пропс: текст выбранного сообщения
    selectedMessage: {
      type: String,
      default: ''
    }
  },
  data() {
    return {
      truncatedText: '',
      summary: ''
    }
  },
  watch: {
    messages(newMessages) {
      if (!this.selectedMessage && newMessages.length > 0) {
        this.getTruncatedText()
      } else if (this.selectedMessage) {
        this.getTruncatedTextFromSelected(this.selectedMessage)
      }
    },
    selectedMessage(newText) {
      if (newText) {
        this.getTruncatedTextFromSelected(newText)
      }
    }
  },
  methods: {
    async getTruncatedText() {
      let text;
      if (this.selectedTopic.startsWith('news')) {
        text = this.messages[0]?.data?.text || ''
      } else {
        text = this.messages.map(m => m.data.text).join('\n')
      }

      try {
        this.truncatedText = await api.truncateText(text)
      } catch (err) {
        console.error('Ошибка при обрезке:', err)
        this.truncatedText = ''
      }
    },
    async getSummary() {
      if (!this.truncatedText) {
        alert('Нет текста для пересказа')
        return
      }

      try {
        this.summary =  await api.getSummary(this.truncatedText, this.selectedTopic)
      } catch (err) {
        console.error('Ошибка при получении пересказа:', err)
        this.summary = 'Не удалось получить пересказ'
      }
    },
    async getTruncatedTextFromSelected(text) {
      try {
        this.truncatedText = await api.truncateText(text)
      } catch (err) {
        console.error('Ошибка при обрезке выделенного сообщения:', err)
        this.truncatedText = ''
      }
    },

    clearSummary() {
      this.truncatedText = ''
      this.summary = ''
    }
  }
}
</script>

<style scoped>
.summary-block {
  margin-top: 2rem;
  background-color: #1f1f1f;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.text-block {
  margin-bottom: 1.5rem;
}

.loading {
  margin-bottom: 1.5rem;
  color: #888;
}

pre {
  background-color: #1a1a1a;
  color: #4caf50;
  font-family: monospace;
  font-size: 0.95rem;
  padding: 1rem;
  border-radius: 8px;
  white-space: pre-wrap;
  word-break: break-word;
}

.btn-summarize {
  display: inline-block;
  margin-top: 1rem;
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

.btn-summarize:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
}

.summary-result {
  margin-top: 1.5rem;
}

.summary-text {
  background-color: #2a2a2a;
  color: #81c784;
}

.truncated-input {
  width: 100%;
  height: 300px;
  background-color: #1a1a1a;
  color: #4caf50;
  font-family: monospace;
  font-size: 0.95rem;
  padding: 1rem;
  border: none;
  border-radius: 8px;
  resize: none;
  white-space: pre-wrap;
  word-break: break-word;
  outline: none;
  transition: background-color 0.3s ease;
}

.truncated-input:focus {
  background-color: #2a2a2a;
}

.btn-clear {
  display: inline-block;
  margin-top: 0.5rem;
  margin-right: 1rem;
  padding: 0.5rem 1rem;
  font-size: 0.9rem;
  color: #ccc;
  background-color: #333;
  border: none;
  border-radius: 8px;
  cursor: pointer;
  transition: all 0.2s ease;
}

.btn-clear:hover {
  background-color: #444;
  color: white;
}
</style>