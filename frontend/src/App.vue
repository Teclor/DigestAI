<template>
  <div id="app">
    <header class="app-header">
      <div class="logo">
        <div class="logo-text">
          <h1>TL;DЯ</h1>
          <h6>Кратко и по делу</h6>
        </div>
      </div>
    </header>

    <main class="app-content">
      <ChatViewer />
    </main>
  </div>
</template>

<script>
import ChatViewer from './views/ChatViewer.vue'

export default {
  components: {
    ChatViewer
  }
}
</script>

<style lang="scss">
.app-header {
  padding: 2rem;
  display: flex;
  justify-content: center;
  align-items: center;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.5);
  text-align: center;

  .logo {
    .logo-text {
      h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: bold;
        background: linear-gradient(45deg, #4caf50, #8e44ad, #3498db, #f1c40f);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        line-height: 1.2;
      }

      p {
        font-size: 1.2rem;
        color: #ccc;
        margin-top: 0.5rem;
      }
    }
  }
}
</style>