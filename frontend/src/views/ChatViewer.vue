<template>
  <div class="tabs-wrapper flex justify-center">
    <el-tabs v-model="activeTab" type="card" class="chat-tabs w-full max-w-4xl">
      <el-tab-pane label="Сообщения" name="messages">
        <OllamaSettings/>
        <MessageList
          :topics="topics"
          :selected-topic="selectedTopic"
          :selected-message="selectedMessage"
          :limit="limit"
          @update:selected-topic="selectedTopic = $event"
          @update:limit="limit = $event"
          :messages="messages"
          @select-message="handleSelectedMessage"
          @load-messages="loadMessages"
        />
      </el-tab-pane>

      <el-tab-pane label="Отправить JSON" name="send-json">
        <ChatForm @submit="sendCustomJson" @initDefaultMessages="initDefaultMessages"/>
      </el-tab-pane>

      <el-tab-pane label="JS-код" name="js-code">
        <JsCodeBlock :topic="selectedTopic" :limit="limit" @copied="onCopy" />
      </el-tab-pane>
    </el-tabs>
  </div>
</template>

<script>
import MessageList from '@/components/MessageList.vue'
import ChatForm from '@/components/ChatForm.vue'
import JsCodeBlock from '@/components/JsCodeBlock.vue'
import api from '@/services/api'
import OllamaSettings from '@/components/OllamaSettings.vue'

export default {
  components: {
    MessageList,
    ChatForm,
    JsCodeBlock,
    OllamaSettings
  },
  data() {
    return {
      activeTab: 'messages',
      topics: [],
      selectedTopic: null,
      limit: 10,
      messages: [],
      selectedMessage: null
    }
  },
  async mounted() {
    try {
      const res = await api.getTopics()
      this.topics = Array.isArray(res.data) ? res.data : [{name: 'news', display_name: 'Новости'}]
      if (this.topics.length > 0) this.selectedTopic = this.topics[0]
    } catch (err) {
      console.error('Ошибка при загрузке топиков', err)
      this.topics = ['news', 'chat']
      this.selectedTopic = this.topics[0]
    }
  },
  methods: {
    handleSelectedMessage(text) {
      this.selectedMessage = text
    },
    async loadMessages({ topic, limit }) {
      try {
        const res = await api.getMessages(topic, limit)
        this.messages = res.data.messages
      } catch (err) {
        alert('Ошибка при получении сообщений: ' + err.message)
      }
    },
    async sendCustomJson({ topic, type, json }) {
      const fullTopicName = `${type}_${topic}`

      let parsedJson
      try {
        parsedJson = JSON.parse(json)
      } catch (e) {
        alert("Некорректный JSON")
        return
      }

      const payload = {
        messages: Array.isArray(parsedJson) ? parsedJson : [parsedJson]
      }

      try {
        await api.sendCustomMessages(fullTopicName, payload.messages)
        alert(`Сообщения успешно отправлены в топик ${fullTopicName}`)
      } catch (err) {
        alert("Ошибка при отправке сообщений: " + (err.response?.data?.detail || err.message))
      }
    },
    async setOllamaUrl({url}) {
      try {
        await api.setOllamaUrl(url)
        console.log(`Хост успешно установлен`)
      } catch (err) {
        alert("Ошибка при установке хоста: " + (err.response?.data?.detail || err.message))
      }
    },
    async initDefaultMessages() {
      if (!confirm("Вы уверены, что хотите заполнить топики данными из файлов?")) {
        return
      }

      try {
        api.initDefaultMessages()
        alert("Данные успешно заполнены!")
      } catch (err) {
        alert("Ошибка при инициализации данных: " + (err.response?.data?.detail || err.message))
      }
    },
    onCopy() {
      alert('JS-код скопирован в буфер обмена!')
    }
  }
}
</script>