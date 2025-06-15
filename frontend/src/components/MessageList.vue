<template>
  <div class="messages-view">
    <!-- Блок с TopicSelector -->
    <div class="selector-wrapper">
      <TopicSelector
          :topics="topics"
          :model-value="selectedTopic"
          :limit="limit"
          @update:model-value="$emit('update:selectedTopic', $event)"
          @update:limit="$emit('update:limit', $event)"
          @load-messages="onLoadMessages"
      />
    </div>

    <!-- Список сообщений -->
    <div v-if="messages.length > 0" class="message-list">
      <div
          v-for="(msg, idx) in messages"
          :key="idx"
          class="message-card"
          @click="selectMessage(msg.data.text)"
      >
        <pre class="message-text">{{ msg.data.text }}</pre>
      </div>
    </div>

    <!-- Сообщение при пустом списке -->
    <div v-else class="empty-state">
      Нет сообщений. Нажмите "Загрузить", чтобы получить последние данные.
    </div>

    <SummaryBlock
        :messages="messages"
        :selectedTopic="selectedTopic"
        :selectedMessage="selectedMessage"
    />
  </div>
</template>

<script>
import TopicSelector from '@/components/TopicSelector.vue'
import SummaryBlock from '@/components/SummaryBlock.vue'

export default {
  components: {
    TopicSelector,
    SummaryBlock
  },
  props: ['topics', 'selectedTopic', 'limit', 'messages', 'selectedMessage'],
  methods: {
    onLoadMessages(data) {
      const payload = data || {topic: this.selectedTopic, limit: this.limit}
      this.$emit('load-messages', payload)
    },
    selectMessage(text) {
      if (this.selectedTopic.startsWith('news')) {
        this.$emit('select-message', text)
      }
    }
  }
}
</script>

<style scoped>
.messages-view {
  background-color: #1a1a1a;
  padding: 2rem;
  border-radius: 16px;
  box-shadow: 0 4px 20px rgba(0, 0, 0, 0.3);
  max-width: 800px;
  margin: 2rem auto;
}

.selector-wrapper {
  margin-bottom: 2rem;
}

.btn-load {
  padding: 0.75rem 1.5rem;
  font-size: 1rem;
  font-weight: bold;
  color: white;
  background: linear-gradient(to right, #0f0c29, #302b63, #24243e);
  border: none;
  border-radius: 10px;
  cursor: pointer;
  transition: all 0.3s ease;
  display: inline-block;
  margin-bottom: 2rem;
}

.btn-load:hover {
  transform: scale(1.03);
  box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
}

.message-list {
  max-height: 500px;
  overflow-y: auto;
  background-color: #222;
  border-radius: 12px;
  padding: 1rem;
  box-shadow: inset 0 0 6px rgba(0, 0, 0, 0.4);
}

.message-card {
  padding: 1rem;
  margin-bottom: 1rem;
  background-color: #1f1f1f;
  border-left: 4px solid #4caf50;
  border-radius: 10px;
  transition: background-color 0.3s ease;
}

.message-card:hover {
  background-color: #2a2a2a;
}

.message-text {
  margin: 0;
  font-family: 'Courier New', Courier, monospace;
  font-size: 1rem;
  color: #4caf50;
  white-space: pre-wrap;
  word-break: break-word;
}

.empty-state {
  text-align: center;
  color: #888;
  font-style: italic;
  margin-top: 2rem;
  padding: 2rem;
  background-color: #151515;
  border-radius: 12px;
  border: 1px dashed #333;
}
</style>