import logging
import traceback

from datetime import timezone
from django.contrib.auth import authenticate
from rest_framework.views import APIView
from rest_framework.response import Response
from rest_framework.permissions import IsAuthenticated
from rest_framework_simplejwt.tokens import RefreshToken
from .models import Message, User
from .serializers import UserSerializer, MessageSerializer
from .bot.utils import StateMachine
from rest_framework.decorators import api_view, permission_classes
from .bot.Memory.LLM_Memory import MemoryManager
from .bot.simple_bot import simple_bot_response
from .bot.placebo_bot import placebo_bot_response

from django.http import JsonResponse
from django.core.files.storage import default_storage
from django.core.files.base import ContentFile
from .bot.ASR.ASRPipeline import feed_audio_to_ASR_modal

# Create shared instances at module level
state_machine = StateMachine()
memory_manager = MemoryManager()

logger = logging.getLogger(__name__)

class RegisterView(APIView):
    def get_balanced_group(self):
        """Get balanced group assignment based on current user counts across all three groups"""
        control_count = User.objects.filter(group='control').count()
        intervention_count = User.objects.filter(group='intervention').count()
        placebo_count = User.objects.filter(group='placebo').count()

        # Find the group with the minimum count
        group_counts = {
            'control': control_count,
            'intervention': intervention_count,
            'placebo': placebo_count
        }

        # Get the group(s) with minimum count
        min_count = min(group_counts.values())
        min_groups = [group for group, count in group_counts.items() if count == min_count]

        # If multiple groups have the same minimum count, prioritize based on:
        # 1. Alpha (intervention) - highest priority
        # 2. Beta (control) - second priority
        # 3. Gamma (placebo) - third priority
        priority_order = ['intervention', 'control', 'placebo']
        for group in priority_order:
            if group in min_groups:
                return group
        
        # Fallback (should never reach here)
        return min_groups[0]

    def post(self, request):
        data = request.data.copy()
        # Assign group if not provided
        if 'group' not in data:
            data['group'] = self.get_balanced_group()
        serializer = UserSerializer(data=data)
        if serializer.is_valid():
            user = serializer.save()
            # Return group in response
            response_data = serializer.data
            response_data['group'] = user.group
            return Response(response_data, status=201)
        return Response(serializer.errors, status=400)


class LoginView(APIView):
    def post(self, request):
        username = request.data.get('username')
        password = request.data.get('password')
        user = authenticate(username=username, password=password)
        if user:
            refresh = RefreshToken.for_user(user)
            return Response({
                'refresh': str(refresh),
                'access': str(refresh.access_token),
                'group': user.group,
            })
        return Response({"error": "Invalid credentials"}, status=400)


class MessageView(APIView):
    permission_classes = [IsAuthenticated]

    def keep_only_numbers(self, s):
        if s is None:
            return None
        return ''.join(char for char in s if char.isdigit())

    def post(self, request):
        try:
            user = request.user
            text = None

            if 'audio' in request.FILES:
                try:
                    audio_file = request.FILES['audio']
                    logger.info(f"Received audio file: {audio_file.name}")
                    
                    file_name = default_storage.save(f'audio/{audio_file.name}', ContentFile(audio_file.read()))
                    file_path = default_storage.path(file_name)
                    
                    logger.info(f"Saved audio file to: {file_path}")
                    
                    text = feed_audio_to_ASR_modal(file_path)
                    logger.info(f"Transcribed text: {text}")
                    
                    default_storage.delete(file_name)
                except Exception as e:
                    logger.error(f"Audio processing error: {str(e)}")
                    logger.error(traceback.format_exc())
                    return Response({
                        "error": f"Error processing audio: {str(e)}",
                        "traceback": traceback.format_exc()
                    }, status=500)
            else:
                text = request.data.get('text')
                
            if not text:
                return Response({
                    "error": "No text or audio input provided"
                }, status=400)
            
            logger.info(f"Processing message from user: {user}")
            
            response_text, recommendations, state, explainibility, exercise_number = state_machine.execute_state(text, user)
            
            if response_text is None:
                return Response({
                    "response": "دارم فکر میکنم ...",
                    "recommendations": [],
                    "state": "PROCESSING",
                    "explainibility": None,
                    "excercise_number": None
                }, status=202)
                
            excercise_number = self.keep_only_numbers(exercise_number)
            
            return Response({
                "response": response_text, 
                "recommendations": recommendations, 
                "state": state, 
                "explainibility": explainibility, 
                "excercise_number": excercise_number
            }, status=200)

        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            logger.error(traceback.format_exc())
            return Response({
                "error": str(e),
                "traceback": traceback.format_exc()
            }, status=500)
class SimpleBotView(APIView):
    permission_classes = [IsAuthenticated]

    def get_current_session_id(self, user):
        """Get the latest session_id for the user, or start a new one if none exists."""
        latest_message = Message.objects.filter(user=user).order_by('-timestamp').first()
        if latest_message:
            return latest_message.session_id
        else:
            return 1  # Start with session 1 if no messages exist

    def get_last_n_history(self, user, n=6):
        """
        Retrieve the last n messages (user and assistant) for the current user.
        Returns a list of dicts: [{"role": "user"/"assistant", "content": ...}, ...]
        """
        messages = (
            Message.objects.filter(user=user)
            .order_by('-timestamp')[:n]
            .values('is_user', 'text')
        )
        # Reverse to chronological order
        messages = list(messages)[::-1]
        history = []
        for msg in messages:
            role = "user" if msg["is_user"] else "assistant"
            history.append({"role": role, "content": msg["text"]})
        return history

    def post(self, request):
        text = request.data.get('text')
        user = request.user

        # Get or create session_id
        session_id = self.get_current_session_id(user)

        # Save user message
        Message.objects.create(user=user, text=text, is_user=True, session_id=session_id)

        # Retrieve last N messages for history (excluding the current user message)
        history = self.get_last_n_history(user, n=6)

        # Call the bot
        response_text, recommendations, updated_history = simple_bot_response(history, text, user)

        # Save assistant message
        Message.objects.create(user=user, text=response_text, is_user=False, session_id=session_id)

        # Optionally, update history with the new messages
        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response_text}
        ]
        return Response({
            "response": response_text,
            "recommendations": recommendations,
            "history": updated_history
        }, status=200)


class PlaceboBotView(APIView):
    permission_classes = [IsAuthenticated]

    def get_current_session_id(self, user):
        """Get the latest session_id for the user, or start a new one if none exists."""
        latest_message = Message.objects.filter(user=user).order_by('-timestamp').first()
        if latest_message:
            return latest_message.session_id
        else:
            return 1  # Start with session 1 if no messages exist

    def get_last_n_history(self, user, n=6):
        """
        Retrieve the last n messages (user and assistant) for the current user.
        Returns a list of dicts: [{"role": "user"/"assistant", "content": ...}, ...]
        """
        messages = (
            Message.objects.filter(user=user)
            .order_by('-timestamp')[:n]
            .values('is_user', 'text')
        )
        # Reverse to chronological order
        messages = list(messages)[::-1]
        history = []
        for msg in messages:
            role = "user" if msg["is_user"] else "assistant"
            history.append({"role": role, "content": msg["text"]})
        return history

    def post(self, request):
        text = request.data.get('text')
        user = request.user

        # Get or create session_id
        session_id = self.get_current_session_id(user)

        # Save user message
        Message.objects.create(user=user, text=text, is_user=True, session_id=session_id)

        # Retrieve last N messages for history (excluding the current user message)
        history = self.get_last_n_history(user, n=6)

        # Call the placebo bot
        response_text, recommendations, updated_history = placebo_bot_response(history, text, user)

        # Save assistant message
        Message.objects.create(user=user, text=response_text, is_user=False, session_id=session_id)

        # Optionally, update history with the new messages
        updated_history = history + [
            {"role": "user", "content": text},
            {"role": "assistant", "content": response_text}
        ]
        return Response({
            "response": response_text,
            "recommendations": recommendations,
            "history": updated_history
        }, status=200)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_chat_history(request):
    session_id = request.GET.get('session_id')
    user = request.user

    if session_id:
        # Get messages for specific session
        messages = memory_manager.get_session_messages(user, int(session_id))
    else:
        # Get messages for current session
        messages = memory_manager.get_current_session_messages(user)

    serializer = MessageSerializer(messages, many=True)
    return Response(serializer.data)


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_user_sessions(request):
    """Get all session IDs for the authenticated user"""
    user = request.user
    from api.models import Message
    from django.db.models import Max

    # Get all unique session IDs for the user
    sessions = Message.objects.filter(user=user).values('session_id').distinct().order_by('session_id')
    session_list = [session['session_id'] for session in sessions]

    return Response({
        'sessions': session_list,
        'total_sessions': len(session_list)
    })


@api_view(['GET'])
@permission_classes([IsAuthenticated])
def get_memory(request):
    memory = memory_manager.get_current_memory(request.user)
    return Response({'memory': memory})

# @api_view(['POST'])
# def add_message(request):
#     # Add user message
#     memory_manager.add_message(
#         user=request.user,
#         text=request.data['message'],
#         is_user=True
#     )

#     # Add LLM response (you'll need to generate this)
#     llm_response = "Your LLM response here"
#     memory_manager.add_message(
#         user=request.user,
#         text=llm_response,
#         is_user=False
#     )

#     return Response({'response': llm_response})


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def end_session(request):
    """Handle browser close or explicit session end"""
    state_machine.handle_session_end(request.user)
    return Response({'status': 'success'}, status=200)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def get_audio_message(request):
    if 'audio' not in request.FILES:
        return JsonResponse({'error': 'No audio file provided'}, status=400)

    audio_file = request.FILES['audio']

    # Save the uploaded file temporarily
    file_name = default_storage.save(f'audio/{audio_file.name}', ContentFile(audio_file.read()))
    file_path = default_storage.path(file_name)

    try:
        # Process the audio file
        transcribed_text = feed_audio_to_ASR_modal(file_path)

        # Clean up the temporary file
        default_storage.delete(file_name)

        return JsonResponse({'transcribed_text': transcribed_text})
    except Exception as e:
        # Clean up the temporary file in case of error
        default_storage.delete(file_name)
        return JsonResponse({'error': str(e)}, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def reset_state_machine(request):
    """Reset the state machine to initial state for the authenticated user"""
    try:
        user = request.user
        reset_data = state_machine.reset_state_machine(user)

        return Response({
            'message': 'State machine reset successfully',
            'new_state': reset_data['state'],
            'message_count': reset_data['message_count'],
            'emotion': reset_data['emotion'],
            'response': reset_data['response'],
            'stage': reset_data['stage'],
            'current_day': reset_data['current_day'],
            'session_id': reset_data['session_id']
        }, status=200)
    except Exception as e:
        return Response({
            'error': f'Failed to reset state machine: {str(e)}'
        }, status=500)


@api_view(['POST'])
@permission_classes([IsAuthenticated])
def process_buffered_messages(request):
    """Process any buffered messages for the user"""
    try:
        user = request.user
        response_text, recommendations, state, explainibility, excercise_number = state_machine.process_buffered_messages(user)

        # Check if there were any buffered messages to process
        if response_text is None:
            return Response({
                "response": "هیچ پیام بافر شده‌ای برای پردازش وجود ندارد",
                "recommendations": [],
                "state": "NO_BUFFERED_MESSAGES",
                "explainibility": None,
                "excercise_number": None
            }, status=200)

        # Keep only numbers for exercise number
        excercise_number = MessageView().keep_only_numbers(excercise_number)

        return Response({
            "response": response_text,
            "recommendations": recommendations,
            "state": state,
            "explainibility": explainibility,
            "excercise_number": excercise_number
        }, status=200)
    except Exception as e:
        return Response({
            'error': f'Failed to process buffered messages: {str(e)}'
        }, status=500)
