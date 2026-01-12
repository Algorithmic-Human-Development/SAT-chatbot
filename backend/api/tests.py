from django.test import TestCase
from django.contrib.auth import get_user_model
from rest_framework.test import APITestCase
from rest_framework import status
from unittest.mock import patch, MagicMock
from .models import User, UserGroup, Message
from .bot.placebo_bot import placebo_bot_response
from .bot.simple_bot import simple_bot_response

User = get_user_model()


class UserGroupTestCase(TestCase):
    """Test the UserGroup enum and model"""

    def test_user_group_choices(self):
        """Test that all three group choices are available"""
        choices = UserGroup.choices()
        expected_choices = [
            ('control', 'CONTROL'),
            ('intervention', 'INTERVENTION'),
            ('placebo', 'PLACEBO')
        ]
        self.assertEqual(choices, expected_choices)

    def test_user_creation_with_placebo_group(self):
        """Test creating a user with placebo group"""
        user = User.objects.create_user(
            username='testuser',
            password='testpass',
            group=UserGroup.PLACEBO
        )
        self.assertEqual(user.group, UserGroup.PLACEBO)


class GroupAssignmentTestCase(APITestCase):
    """Test group assignment logic in registration"""

    def test_balanced_group_assignment(self):
        """Test that users are assigned to groups in a balanced way"""
        # Create some existing users to test balancing
        User.objects.create_user(username='user1', group='control')
        User.objects.create_user(username='user2', group='intervention')

        # Register a new user - should get placebo (smallest group)
        response = self.client.post('/api/register/', {
            'username': 'newuser',
            'password': 'testpass123',
            'email': 'test@example.com'
        })

        self.assertEqual(response.status_code, status.HTTP_201_CREATED)
        user = User.objects.get(username='newuser')
        self.assertEqual(user.group, 'placebo')


class PlaceboBotTestCase(TestCase):
    """Test the placebo bot functionality"""

    def setUp(self):
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass',
            group=UserGroup.PLACEBO
        )

    @patch('api.bot.placebo_bot.openai_req_with_history')
    @patch('api.bot.placebo_bot.create_recommendations')
    def test_placebo_bot_response(self, mock_recommendations, mock_openai):
        """Test placebo bot response generation"""
        # Mock the OpenAI response
        mock_openai.return_value = "سلام، چطور می‌تونم کمکتون کنم؟"
        mock_recommendations.return_value = ["پیشنهاد 1", "پیشنهاد 2", "پیشنهاد 3"]

        history = []
        user_message = "سلام"

        response, recommendations, updated_history = placebo_bot_response(
            history, user_message, self.user
        )

        # Check that OpenAI was called with the right system prompt
        mock_openai.assert_called_once()
        call_args = mock_openai.call_args[0][0]  # Get the messages argument

        self.assertEqual(call_args[0]['role'], 'system')
        self.assertIn('دستیار دلبستگی به خود', call_args[0]['content'])

        # Check response
        self.assertEqual(response, "سلام، چطور می‌تونم کمکتون کنم؟")
        self.assertEqual(len(updated_history), 2)  # user + assistant message

    def test_placebo_bot_with_history(self):
        """Test placebo bot handles conversation history correctly"""
        history = [
            {"role": "user", "content": "سلام"},
            {"role": "assistant", "content": "سلام، حالت چطوره؟"}
        ]

        with patch('api.bot.placebo_bot.openai_req_with_history') as mock_openai, \
                patch('api.bot.placebo_bot.create_recommendations') as mock_recommendations:

            mock_openai.return_value = "خوشحالم که باهام صحبت می‌کنی"
            mock_recommendations.return_value = []

            response, recommendations, updated_history = placebo_bot_response(
                history, "ممنون", self.user
            )

            # Check that history was included in the request
            call_args = mock_openai.call_args[0][0]
            self.assertEqual(len(call_args), 4)  # system + 2 history + new user message


class ChatbotEndpointsTestCase(APITestCase):
    """Test all three chatbot endpoints"""

    def setUp(self):
        # Create users for each group
        self.intervention_user = User.objects.create_user(
            username='intervention_user',
            password='testpass',
            group=UserGroup.INTERVENTION
        )
        self.control_user = User.objects.create_user(
            username='control_user',
            password='testpass',
            group=UserGroup.CONTROL
        )
        self.placebo_user = User.objects.create_user(
            username='placebo_user',
            password='testpass',
            group=UserGroup.PLACEBO
        )

    def test_all_endpoints_exist(self):
        """Test that all three chatbot endpoints exist"""
        # Test intervention endpoint
        self.client.force_authenticate(user=self.intervention_user)
        response = self.client.post('/api/message/', {'text': 'test'})
        self.assertNotEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Test control endpoint
        self.client.force_authenticate(user=self.control_user)
        response = self.client.post('/api/simple-chat/', {'text': 'test'})
        self.assertNotEqual(response.status_code, status.HTTP_404_NOT_FOUND)

        # Test placebo endpoint
        self.client.force_authenticate(user=self.placebo_user)
        response = self.client.post('/api/placebo-chat/', {'text': 'test'})
        self.assertNotEqual(response.status_code, status.HTTP_404_NOT_FOUND)

    @patch('api.bot.placebo_bot.openai_req_with_history')
    @patch('api.bot.placebo_bot.create_recommendations')
    def test_placebo_chat_endpoint(self, mock_recommendations, mock_openai):
        """Test the placebo chat endpoint works correctly"""
        mock_openai.return_value = "پاسخ تست"
        mock_recommendations.return_value = []

        self.client.force_authenticate(user=self.placebo_user)
        response = self.client.post('/api/placebo-chat/', {'text': 'سلام'})

        self.assertEqual(response.status_code, status.HTTP_200_OK)
        self.assertIn('response', response.data)
        self.assertIn('recommendations', response.data)
        self.assertIn('history', response.data)

        # Check that message was saved to database
        message = Message.objects.filter(user=self.placebo_user).first()
        self.assertIsNotNone(message)
        self.assertEqual(message.text, 'سلام')


class ComparisonTestCase(TestCase):
    """Test to ensure different bots work differently"""

    def setUp(self):
        self.user = User.objects.create_user(username='testuser', password='testpass')

    @patch('api.bot.simple_bot.openai_req_with_history')
    @patch('api.bot.placebo_bot.openai_req_with_history')
    def test_different_prompts(self, mock_placebo_openai, mock_simple_openai):
        """Test that simple bot and placebo bot use different system prompts"""
        mock_placebo_openai.return_value = "placebo response"
        mock_simple_openai.return_value = "simple response"

        # Test placebo bot
        placebo_bot_response([], "test", self.user)
        placebo_call_args = mock_placebo_openai.call_args[0][0]

        # Test simple bot (need to mock additional functions)
        with patch('api.bot.simple_bot.get_daily_exercises') as mock_exercises, \
                patch('api.bot.simple_bot.load_sat_knowledge') as mock_knowledge, \
                patch('api.bot.simple_bot.load_system_prompt') as mock_prompt:

            mock_exercises.return_value = "تمرین تست"
            mock_knowledge.return_value = "دانش SAT"
            mock_prompt.return_value = "پرامپت پیشرفته: {daily_exercises} {memory}"

            simple_bot_response([], "test", self.user)
            simple_call_args = mock_simple_openai.call_args[0][0]

        # Compare system prompts
        placebo_system_prompt = placebo_call_args[0]['content']
        simple_system_prompt = simple_call_args[0]['content']

        # Placebo should be simple
        self.assertIn('دستیار دلبستگی به خود', placebo_system_prompt)
        self.assertNotIn('SAT', placebo_system_prompt)

        # Simple should be more complex
        self.assertIn('تمرین تست', simple_system_prompt)
        self.assertIn('دانش SAT', simple_system_prompt)
