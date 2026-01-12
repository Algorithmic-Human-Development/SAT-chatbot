#!/usr/bin/env python3
"""
Test script for message buffering functionality
This script tests the message buffering system to ensure it properly handles rapid successive messages.
"""

import sys
import os
import django
from django.test import TestCase
from django.contrib.auth import get_user_model
from unittest.mock import Mock, patch

# Add the backend directory to the Python path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Setup Django
os.environ.setdefault('DJANGO_SETTINGS_MODULE', 'backend.settings')
django.setup()

from api.bot.utils import StateMachine, MessageBuffer

class MessageBufferingTest(TestCase):
    """Test cases for message buffering functionality"""
    
    def setUp(self):
        """Set up test environment"""
        self.state_machine = StateMachine()
        self.message_buffer = MessageBuffer()
        
        # Create a mock user
        User = get_user_model()
        self.user = User.objects.create_user(
            username='testuser',
            password='testpass123',
            email='test@example.com'
        )
    
    def test_message_buffer_initialization(self):
        """Test that message buffer initializes correctly"""
        self.assertIsNotNone(self.message_buffer.processing_users)
        self.assertIsNotNone(self.message_buffer.message_buffers)
        self.assertIsNotNone(self.message_buffer.lock)
    
    def test_user_processing_status(self):
        """Test user processing status tracking"""
        user_id = self.user.id
        
        # Initially, user should not be processing
        self.assertFalse(self.message_buffer.is_user_processing(user_id))
        
        # Start processing
        self.message_buffer.start_processing(user_id)
        self.assertTrue(self.message_buffer.is_user_processing(user_id))
        
        # End processing
        self.message_buffer.end_processing(user_id)
        self.assertFalse(self.message_buffer.is_user_processing(user_id))
    
    def test_message_buffering(self):
        """Test that messages are properly buffered"""
        user_id = self.user.id
        
        # Start processing
        self.message_buffer.start_processing(user_id)
        
        # Add messages while processing
        self.message_buffer.add_message(user_id, "پیام اول")
        self.message_buffer.add_message(user_id, "پیام دوم")
        self.message_buffer.add_message(user_id, "پیام سوم")
        
        # Check that messages are buffered
        self.assertTrue(self.message_buffer.has_buffered_messages(user_id))
        
        # Get buffered messages
        buffered_messages = self.message_buffer.get_buffered_messages(user_id)
        self.assertEqual(len(buffered_messages), 3)
        self.assertEqual(buffered_messages[0], "پیام اول")
        self.assertEqual(buffered_messages[1], "پیام دوم")
        self.assertEqual(buffered_messages[2], "پیام سوم")
        
        # Buffer should be cleared after getting messages
        self.assertFalse(self.message_buffer.has_buffered_messages(user_id))
        
        # End processing
        self.message_buffer.end_processing(user_id)
    
    def test_message_concatenation(self):
        """Test message concatenation functionality"""
        messages = ["سلام", "چطوری؟", "خوبم ممنون"]
        concatenated = self.message_buffer.concatenate_messages(messages)
        expected = "سلام چطوری؟ خوبم ممنون"
        self.assertEqual(concatenated, expected)
    
    def test_empty_message_concatenation(self):
        """Test concatenation with empty messages"""
        # Empty list
        result = self.message_buffer.concatenate_messages([])
        self.assertEqual(result, "")
        
        # List with empty strings
        result = self.message_buffer.concatenate_messages(["", "سلام", ""])
        self.assertEqual(result, "سلام")
    
    def test_concurrent_access(self):
        """Test that the buffer handles concurrent access safely"""
        import threading
        import time
        
        user_id = self.user.id
        results = []
        
        def add_message(message):
            time.sleep(0.01)  # Small delay to simulate concurrent access
            self.message_buffer.add_message(user_id, message)
            results.append(f"Added: {message}")
        
        # Start processing
        self.message_buffer.start_processing(user_id)
        
        # Create multiple threads to add messages concurrently
        threads = []
        for i in range(5):
            thread = threading.Thread(target=add_message, args=(f"پیام {i}",))
            threads.append(thread)
            thread.start()
        
        # Wait for all threads to complete
        for thread in threads:
            thread.join()
        
        # Check that all messages were added
        buffered_messages = self.message_buffer.get_buffered_messages(user_id)
        self.assertEqual(len(buffered_messages), 5)
        
        # End processing
        self.message_buffer.end_processing(user_id)
    
    def test_state_machine_buffering_integration(self):
        """Test integration with state machine"""
        user_id = self.user.id
        
        # Mock the state machine's processing methods to avoid actual LLM calls
        with patch.object(self.state_machine, 'state_handler') as mock_handler:
            mock_handler.return_value = ("پاسخ تست", [], None, None)
            
            # First message should start processing
            result1 = self.state_machine.execute_state("پیام اول", self.user)
            self.assertIsNotNone(result1[0])  # Should return a response
            
            # Second message while processing should be buffered
            result2 = self.state_machine.execute_state("پیام دوم", self.user)
            self.assertIsNone(result2[0])  # Should return None (buffered)
            
            # Third message while processing should also be buffered
            result3 = self.state_machine.execute_state("پیام سوم", self.user)
            self.assertIsNone(result3[0])  # Should return None (buffered)
    
    def tearDown(self):
        """Clean up after tests"""
        # Clean up any remaining processing status
        self.message_buffer.end_processing(self.user.id)
        
        # Delete test user
        self.user.delete()

if __name__ == '__main__':
    # Run the tests
    import unittest
    unittest.main()
