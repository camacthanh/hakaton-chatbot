from flask import Flask, request, jsonify, render_template_string
from rag_traffic_law_bot import app as rag_app  # Import the LangGraph app
from langchain_core.messages import HumanMessage, AIMessage
import os
import uuid

app = Flask(__name__)

# In-memory session storage for chat history
SESSIONS = {}

# ==============================
#  HTML + JS UI
# ==============================

HTML_PAGE = r"""
{% raw %}
<!DOCTYPE html>
<html lang="vi">
<head>
  <meta charset="UTF-8" />
  <title>Traffic Law Assistant - Chat</title>
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <style>
  :root {
    --bg: #f3f4f6;
    --bg-elevated: #ffffff;
    --bg-sidebar: #f9fafb;
    --accent: #16a34a;
    --text: #111827;
    --text-soft: #6b7280;
    --border: #e5e7eb;
    --user-bubble: #1d4ed8;
  }
  body { margin: 0; font-family: system-ui, sans-serif; background: var(--bg); color: var(--text); display: flex; flex-direction: column; }
  .main-pane { flex: 1; display: flex; flex-direction: column; padding: 12px; max-width: 980px; margin: 0 auto; width: 100%; }
  .app-header { padding: 12px 16px; border-radius: 16px; background: #fff; border: 1px solid var(--border); box-shadow: 0 8px 20px rgba(15, 23, 42, 0.06); }
  .app-title h1 { font-size: 18px; margin: 0; }
  .app-title p { margin: 0; font-size: 12px; color: var(--text-soft); }
  .chat-container { flex: 1; margin-top: 10px; border-radius: 16px; background: #fff; border: 1px solid var(--border); box-shadow: 0 4px 18px rgba(148, 163, 184, 0.15); display: flex; flex-direction: column; overflow: hidden; }
  .chat-messages { flex: 1; padding: 16px; overflow-y: auto; }
  .message-row { display: flex; margin-bottom: 12px; }
  .message-row.user { justify-content: flex-end; }
  .message-row.assistant { justify-content: flex-start; }
  .message-bubble { max-width: 80%; padding: 10px 12px; border-radius: 16px; font-size: 14px; line-height: 1.5; word-wrap: break-word; border: 1px solid #e5e7eb; }
  .message-row.user .message-bubble { background: linear-gradient(135deg, #2563eb, #22c55e); color: #f9fafb; border-radius: 16px 4px 16px 16px; }
  .message-row.assistant .message-bubble { background: #f9fafb; color: var(--text); border-radius: 4px 16px 16px 16px; }
  .chat-footer { padding: 10px; border-top: 1px solid var(--border); background: #fff; }
  .input-row { display: flex; gap: 8px; align-items: flex-end; }
  textarea#user-input { flex: 1; resize: none; min-height: 44px; border-radius: 12px; border: 1px solid #d1d5db; padding: 8px 10px; font-size: 14px; }
  button#send-btn { border: none; border-radius: 999px; padding: 10px 16px; background: linear-gradient(135deg, #22c55e, #16a34a); color: #f9fafb; font-weight: 600; cursor: pointer; }
  button#send-btn:disabled { opacity: 0.6; cursor: not-allowed; }
  </style>
</head>
<body>
  <div class="main-pane">
    <header class="app-header">
      <div class="app-title">
        <h1>Traffic Law Assistant</h1>
        <p>Your AI-powered legal assistant for Vietnamese traffic law.</p>
      </div>
    </header>
    <main class="chat-container">
      <div id="chat-messages" class="chat-messages"></div>
      <footer class="chat-footer">
        <div class="input-row">
          <textarea id="user-input" rows="1" placeholder="Nhập câu hỏi..."></textarea>
          <button id="send-btn">Gửi</button>
        </div>
      </footer>
    </main>
  </div>
  <script>
    const chatMessagesEl = document.getElementById("chat-messages");
    const userInputEl = document.getElementById("user-input");
    const sendBtnEl = document.getElementById("send-btn");
    let sessionId = "session_" + Math.random().toString(36).substr(2, 9);
    let isSending = false;

    function addMessageToDOM(role, text) {
      const row = document.createElement("div");
      row.className = "message-row " + role;
      const bubble = document.createElement("div");
      bubble.className = "message-bubble";
      bubble.textContent = text;
      row.appendChild(bubble);
      chatMessagesEl.appendChild(row);
      chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
      return row;
    }

    async function sendMessage() {
      if (isSending) return;
      const text = userInputEl.value.trim();
      if (!text) return;

      addMessageToDOM("user", text);
      userInputEl.value = "";
      const thinkingRow = addMessageToDOM("assistant", "Đang suy nghĩ...");
      setSending(true);

      try {
        const response = await fetch("/api/chat", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ message: text, session_id: sessionId }),
        });
        const data = await response.json();
        thinkingRow.querySelector(".message-bubble").textContent = data.reply;
      } catch (err) {
        thinkingRow.querySelector(".message-bubble").textContent = "Lỗi: " + err.message;
      } finally {
        setSending(false);
      }
    }
    
    function setSending(state) {
        isSending = state;
        sendBtnEl.disabled = state;
        userInputEl.disabled = state;
    }

    userInputEl.addEventListener("keydown", (e) => {
      if (e.key === "Enter" && !e.shiftKey) {
        e.preventDefault();
        sendMessage();
      }
    });
    sendBtnEl.addEventListener("click", sendMessage);
    addMessageToDOM("assistant", "Xin chào! Tôi là trợ lý pháp lý giao thông. Bạn có thể hỏi tôi về luật giao thông Việt Nam.");
  </script>
</body>
</html>
{% endraw %}
"""

# ==============================
#  ROUTES
# ==============================

@app.route("/")
def index():
    return render_template_string(HTML_PAGE)

@app.route("/api/chat", methods=["POST"])
def chat():
    data = request.get_json(silent=True) or {}
    message = data.get("message", "").strip()
    session_id = data.get("session_id")

    if not message or not session_id:
        return jsonify({"reply": "Thiếu thông tin cần thiết."}), 400

    if session_id not in SESSIONS:
        SESSIONS[session_id] = []
    
    chat_history = SESSIONS[session_id]
    
    MAX_HISTORY_TURNS = 3
    truncated_history = chat_history[-(MAX_HISTORY_TURNS * 2):]

    try:
        initial_state = {"question": message, "chat_history": truncated_history}
        final_state = rag_app.invoke(initial_state)
        reply = final_state.get("generation", "Không có câu trả lời được tạo ra.")
        
        chat_history.extend([HumanMessage(content=message), AIMessage(content=reply)])
        return jsonify({"reply": reply})
    except Exception as e:
        print(f"Error during RAG invocation: {e}")
        return jsonify({"reply": f"Lỗi server: {str(e)}"}), 500

# This is the entry point for Vercel
if __name__ == "__main__":
    # This block is for local development and will not be run on Vercel
    app.run(debug=True)
