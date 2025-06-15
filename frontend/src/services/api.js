// src/services/api.js
import axios from 'axios'

const apiClient = axios.create({
  baseURL: '/api'
})

export default {
  getTopics() {
    return apiClient.get('/topics')
  },
  getMessages(topic, limit) {
    return apiClient.get('/messages', { params: { topic, limit } })
  },
  initDefaultMessages() {
    return apiClient.post('/messages/init')
  },
  sendCustomMessages(topic, payload) {
    return apiClient.post('/messages/add', payload, { params: { topic } })
  },
  async truncateText(text, maxTokens = 200) {
    const response = await axios.post('/api/truncated', {text})
    return response.data
  },
  async getSummary(text, topic) {
    const response = await axios.post('/api/summary', { text, topic })
    return response.data.summary
  },
  async getOllamaUrl() {
    const res = await axios.get('/api/ollama/url')
    return res.data
  },

  async setOllamaUrl(url) {
    const res = await axios.post('/api/ollama/url', { url: url })
    return res.data
  },

  async getOllamaModels() {
    const res = await axios.get('/api/ollama/models')
    return res.data
  },

  async setOllamaModel(model) {
    const res = await axios.post('/api/ollama/model', { model })
    return res.data
  }
}