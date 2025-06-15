<template>
  <div class="form-container">
    <h3>Отправить JSON в топик</h3>

    <div class="form-group">
      <label>ID Чата:</label>
      <input v-model="topic" placeholder="Например: chat_123" />
    </div>

    <div class="form-group">
      <label>Тип:</label>
      <select v-model="type">
        <option value="news">news</option>
        <option value="chat">chat</option>
      </select>
    </div>

    <div class="form-group">
      <label>JSON сообщение:</label>
      <textarea v-model="json" rows="4" placeholder='[{"text": "Привет!"}]'></textarea>
    </div>

    <button @click="submit">Отправить</button>

    <div style="margin-top: 2rem;">
      <button @click="initDefaultMessages">Заполнить по умолчанию</button>
    </div>

  </div>
</template>

<script>
export default {
  data() {
    return {
      topic: '',
      type: 'chat',
      json: ''
    }
  },
  methods: {
    submit() {
      this.$emit('submit', {
        topic: this.topic,
        type: this.type,
        json: this.json
      })
    },
    initDefaultMessages() {
      this.$emit('init-default-messages');
    }
  }
}
</script>

<style scoped>
.form-container {
  background-color: #1a1a1a;
  padding: 2rem;
  border-radius: 12px;
  margin: 1rem 0;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.4);
}

.form-group {
  margin-bottom: 1.5rem;
}

label {
  display: block;
  margin-bottom: 0.75rem;
  font-weight: bold;
  font-size: 1.1rem;
}

input,
select,
textarea {
  width: 100%;
  box-sizing: border-box;
  background-color: #2a2a2a;
  color: #fff;
  border: 1px solid #444;
  font-size: 1rem;
  padding: 0.75rem;
}

button {
  padding: 0.75rem 2rem;
  font-size: 1.1rem;
  background-color: #4caf50;
  color: white;
  border: none;
  border-radius: 6px;
  cursor: pointer;
}

button:hover {
  background-color: #66bb6a;
}
</style>