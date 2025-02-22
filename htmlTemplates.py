css = '''
<style>
.chat-message {
    padding: 1.5rem; border-radius: 0.5rem; margin-bottom: 1rem; display: flex
}
.chat-message.user {
    background-color: #2b313e
}
.chat-message.bot {
    background-color: #475063
}
.chat-message .avatar {
  width: 20%;
}
.chat-message .avatar img {
  max-width: 78px;
  max-height: 78px;
  border-radius: 50%;
  object-fit: cover;
}
.chat-message .message {
  width: 80%;
  padding: 0 1.5rem;
  color: #fff;
}
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="http://www.campustimes.org/wp-content/uploads/2020/12/Chatbot_Bridget_Tokiwa-800x560-c-default.png" style="max-height: 78px; max-width: 78px; border-radius: 50%; object-fit: cover;">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://static.vecteezy.com/system/resources/previews/023/784/016/non_2x/a-man-asks-a-question-to-artificial-intelligence-bot-chatbot-in-the-form-of-a-cute-robot-answers-questions-ai-and-human-characters-using-and-chatting-messanger-neural-network-conversation-vector.jpg">
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
