from django.urls import path
from .views import RegisterView, LoginView, MessageView, end_session, get_audio_message
from .views import SimpleBotView, PlaceboBotView, reset_state_machine, get_chat_history, get_user_sessions, process_buffered_messages

urlpatterns = [
    path('register/', RegisterView.as_view(), name='register'),
    path('login/', LoginView.as_view(), name='login'),
    path('message/', MessageView.as_view(), name='message'),
    path('end-session/', end_session, name='end-session'),
    path('send-audio/', get_audio_message, name='send-audio'),
    path('simple-chat/', SimpleBotView.as_view(), name='simple-chat'),
    path('placebo-chat/', PlaceboBotView.as_view(), name='placebo-chat'),
    path('reset-state/', reset_state_machine, name='reset-state'),
    path('process-buffered/', process_buffered_messages, name='process-buffered'),
    path('chat-history/', get_chat_history, name='chat-history'),
    path('user-sessions/', get_user_sessions, name='user-sessions'),
]
