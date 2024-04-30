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

# Bot의 대화창 디자인
bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://cdn-icons-png.flaticon.com/512/7364/7364323.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

# user dialog window design
user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://www.pork.com.au/wp-content/uploads/sites/3/2021/03/Question-Mark-FAQ-Website.png">
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''