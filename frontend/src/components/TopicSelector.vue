<template>
  <div class="topic-selector">
    <!-- Выбор топика -->
    <div class="form-group">
      <label>Выберите чат:</label>
      <select
        :value="modelValue"
        @input="$emit('update:modelValue', $event.target.value)"
        class="input-field"
      >
        <option v-for="topic in topics" :key="topic.name" :value="topic.name">
          {{ topic.display_name }}
        </option>
      </select>
    </div>

    <!-- Количество сообщений + кнопка -->
    <div class="form-group form-row">
      <div class="flex-grow">
        <label>Количество сообщений:</label>
        <input
          type="number"
          :value="limit"
          @input="$emit('update:limit', parseInt($event.target.value) || 1)"
          min="1"
          max="1000"
          class="input-field"
        />
      </div>
      <button class="btn-load" @click="loadMessages">Показать</button>
    </div>
  </div>
</template>

<script>
export default {
  props: {
    topics: Array,
    modelValue: [String, Number],
    limit: {
      type: Number,
      default: 10
    }
  },
  emits: ['update:modelValue', 'update:limit', 'load-messages'],
  methods: {
    loadMessages() {
      this.$emit('load-messages', {
        topic: this.modelValue,
        limit: this.limit
      })
    }
  }
}
</script>

<style scoped>
.topic-selector {
  background-color: #1a1a1a;
  padding: 1.5rem;
  border-radius: 12px;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.4);
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

/* Горизонтальная группа: поле + кнопка */
.form-row {
  display: flex;
  gap: 1rem;
  align-items: flex-end;
}

.flex-grow {
  flex: 1;
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
}

.btn-load:hover {
  transform: scale(1.05);
  box-shadow: 0 4px 12px rgba(0, 255, 0, 0.2);
}
</style>