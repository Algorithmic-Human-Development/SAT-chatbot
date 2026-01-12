from api.models import Message, UserMemoryState, UserDayProgress
from api.bot.gpt import openai_req_generator
from api.bot.gpt_for_statedetection import if_data_sufficient_for_state_change
from api.bot.Memory.LLM_Memory import MemoryManager
from api.bot.gpt_for_comprehension import OpenAILLM
from api.bot.gpt_recommendations import create_recommendations
from api.bot.RAG.llm_excercise_suggestor import suggest_exercises, exercises
from api.bot.simple_bot import get_daily_exercises
from api.bot.RAG.gpt_explainability import create_exercise_explanation
import json
import re
import time
import threading
from django.db.models import Max


class RepetitionPrevention:
    """Global repetition prevention system that tracks all phrases used across the entire conversation"""

    def __init__(self):
        self.used_phrases = set()
        self.used_questions = set()
        self.used_empathy_phrases = set()
        self.used_transitions = set()
        self.used_words = {}  # Track word frequency
        self.problematic_words = {
            'کمک': 0,
            'متاسفم': 0,
            'می‌تونم': 0,
            'می‌خوام': 0,
            'اینجام': 0,
            'گوش': 0,
            'صحبت': 0,
            'احساس': 0,
            'ناراحت': 0,
            'خوشحال': 0,
            'واقعاً': 0,
            'وای': 0,
            'اوه': 0,
            'باشه': 0,
            'مشکلی': 0,
            'احترام': 0,
            'دوست': 0,
            'طبیعی': 0,
            'گرم': 0,
            'صمیمی': 0
        }

    def add_phrase(self, phrase, category="general"):
        """Add a phrase to the used phrases set and track word frequency"""
        if phrase:
            # Clean and normalize the phrase
            cleaned_phrase = self._clean_phrase(phrase)
            if cleaned_phrase:
                self.used_phrases.add(cleaned_phrase)

                # Track word frequency
                self._track_word_frequency(cleaned_phrase)

                # Categorize phrases for better tracking
                if category == "question":
                    self.used_questions.add(cleaned_phrase)
                elif category == "empathy":
                    self.used_empathy_phrases.add(cleaned_phrase)
                elif category == "transition":
                    self.used_transitions.add(cleaned_phrase)

    def _track_word_frequency(self, phrase):
        """Track frequency of problematic words"""
        words = phrase.split()
        for word in words:
            if word in self.problematic_words:
                self.problematic_words[word] += 1
            if word in self.used_words:
                self.used_words[word] += 1
            else:
                self.used_words[word] = 1

    def get_overused_words(self, threshold=2):
        """Get words that have been used too frequently"""
        overused = {}
        for word, count in self.problematic_words.items():
            if count >= threshold:
                overused[word] = count
        return overused

    def is_word_overused(self, word, threshold=2):
        """Check if a word has been used too frequently"""
        return self.problematic_words.get(word, 0) >= threshold

    def _clean_phrase(self, phrase):
        """Clean and normalize a phrase for comparison"""
        if not phrase:
            return ""

        # Remove extra whitespace and normalize
        cleaned = re.sub(r'\s+', ' ', phrase.strip())
        # Remove punctuation for comparison
        cleaned = re.sub(r'[؟،.!?]', '', cleaned)
        return cleaned.lower()

    def is_phrase_used(self, phrase, category="general"):
        """Check if a phrase has been used before"""
        if not phrase:
            return False

        cleaned_phrase = self._clean_phrase(phrase)
        if category == "question":
            return cleaned_phrase in self.used_questions
        elif category == "empathy":
            return cleaned_phrase in self.used_empathy_phrases
        elif category == "transition":
            return cleaned_phrase in self.used_transitions
        else:
            return cleaned_phrase in self.used_phrases

    def get_unused_phrases(self, phrase_list, category="general"):
        """Get phrases from a list that haven't been used yet"""
        return [phrase for phrase in phrase_list if not self.is_phrase_used(phrase, category)]

    def reset_for_user(self, user_id):
        """Reset repetition prevention for a specific user (global for now)"""
        self.used_phrases.clear()
        self.used_questions.clear()
        self.used_empathy_phrases.clear()
        self.used_transitions.clear()
        self.used_words.clear()
        for word in self.problematic_words:
            self.problematic_words[word] = 0


class MessageBuffer:
    """Handles buffering of rapid successive messages from users"""

    def __init__(self):
        self.processing_users = {}  # user_id -> processing status
        self.message_buffers = {}   # user_id -> list of buffered messages
        self.lock = threading.Lock()

    def is_user_processing(self, user_id):
        """Check if a user is currently being processed"""
        with self.lock:
            return user_id in self.processing_users and self.processing_users[user_id]

    def start_processing(self, user_id):
        """Mark that processing has started for a user"""
        with self.lock:
            self.processing_users[user_id] = True
            if user_id not in self.message_buffers:
                self.message_buffers[user_id] = []

    def end_processing(self, user_id):
        """Mark that processing has ended for a user"""
        with self.lock:
            self.processing_users[user_id] = False
            # Clear the buffer after processing
            if user_id in self.message_buffers:
                self.message_buffers[user_id] = []

    def add_message(self, user_id, message):
        """Add a message to the buffer for a user"""
        with self.lock:
            if user_id not in self.message_buffers:
                self.message_buffers[user_id] = []
            self.message_buffers[user_id].append(message)

    def get_buffered_messages(self, user_id):
        """Get all buffered messages for a user and clear the buffer"""
        with self.lock:
            if user_id in self.message_buffers:
                messages = self.message_buffers[user_id].copy()
                self.message_buffers[user_id] = []
                return messages
            return []

    def concatenate_messages(self, messages):
        """Concatenate multiple messages into a single message"""
        if not messages:
            return ""
        # Join messages with space, but handle cases where messages might already have punctuation
        concatenated = " ".join(messages)
        # Clean up any double spaces
        concatenated = re.sub(r'\s+', ' ', concatenated).strip()
        return concatenated

    def has_buffered_messages(self, user_id):
        """Check if there are any buffered messages for a user"""
        with self.lock:
            return user_id in self.message_buffers and len(self.message_buffers[user_id]) > 0


class StateMachine:
    def __init__(self):
        self.user_states = {}
        self.memory_manager = MemoryManager()
        self.openai_llm = OpenAILLM()
        self.repetition_prevention = RepetitionPrevention()
        self.message_buffer = MessageBuffer()

        # Predefined phrase banks for variety
        self.empathy_phrases = [
            "وای، این واقعاً سخته...",
            "می‌تونم بفهمم چه احساسی داری",
            "این اتفاق تأثیر زیادی رویت گذاشته",
            "واقعاً ناراحت‌کننده‌ست...",
            "این تجربه سختی بوده برات",
            "می‌فهمم که چقدر سخت بوده",
            "این موضوع واقعاً مهمه",
            "وای، این واقعاً دردناک بوده...",
            "می‌تونم تصور کنم چه احساسی داری",
            "این تجربه واقعاً دشوار بوده",
            "این اتفاق تأثیر عمیقی رویت گذاشته",
            "واقعاً ناراحت‌کننده‌ست...",
            "می‌فهمم که چقدر سخت بوده برات",
            "این تجربه واقعاً دردناک بوده",
            "این موضوع واقعاً ناراحت‌کننده‌ست",
            "می‌فهمم که چقدر دشوار بوده",
            "این تجربه واقعاً تأثیرگذار بوده",
            "وای، این واقعاً سخته...",
            "می‌تونم بفهمم که چقدر ناراحت‌کننده‌ست",
            "این تجربه سختی بوده برات",
            "امیدوارم بتونم کمکت کنم",
            "اگر نیاز به صحبت داری، من اینجا هستم",
            "این اتفاق واقعاً ناراحت‌کننده‌ست",
            "می‌خوام بدونم چطور کمکت کنم",
            "اگر دوست داری، می‌تونی بیشتر بگی",
            "امیدوارم حالت بهتر بشه",
            "این تجربه سختی بوده",
            "می‌فهمم چه احساسی داری",
            "حق با شماست",
            "این احساس طبیعی است",
            "کاملاً درکت می‌کنم",
            "این تجربه واقعاً ناراحت‌کننده‌ست",
            "می‌تونم بفهمم که چقدر سخت بوده",
            "این اتفاق تأثیر عمیقی روی زندگی‌ات گذاشته",
            "می‌فهمم که این تجربه چقدر می‌تونه سخت باشه",
            "این موضوع واقعاً ناراحت‌کننده‌ست",
            "می‌تونم تصور کنم چه احساسی داری",
            "این تجربه واقعاً دشوار بوده",
            "این اتفاق تأثیر زیادی رویت گذاشته",
            "می‌فهمم که چقدر سخت بوده برات",
            "این تجربه واقعاً دردناک بوده",
            "این موضوع واقعاً ناراحت‌کننده‌ست",
        ]

        # Alternative phrases to avoid repetition
        self.alternative_help_phrases = [
            "اگر دوست داری، می‌تونی بیشتر بگی",
            "من اینجا هستم تا گوش بدم",
            "اگر نیاز به صحبت داری، بگو",
            "می‌تونی بیشتر توضیح بدی",
            "اگر دوست داری، ادامه بده",
            "من گوش می‌کنم",
            "اگر نیاز داری، بگو",
            "می‌تونی بیشتر بگی",
            "اگر دوست داری، ادامه بده",
            "من اینجا هستم",
            "اگر نیاز داری، بگو",
            "می‌تونی بیشتر توضیح بدی",
            "اگر دوست داری، بگو",
            "من گوش می‌کنم",
            "اگر نیاز داری، ادامه بده",
        ]

        self.alternative_empathy_phrases = [
            "این تجربه سختی بوده",
            "این اتفاق تأثیر زیادی رویت گذاشته",
            "این موضوع واقعاً مهمه",
            "این تجربه واقعاً دشوار بوده",
            "این اتفاق تأثیر عمیقی رویت گذاشته",
            "این تجربه واقعاً دردناک بوده",
            "این موضوع واقعاً ناراحت‌کننده‌ست",
            "این تجربه واقعاً تأثیرگذار بوده",
            "این تجربه سختی بوده برات",
            "این اتفاق تأثیر زیادی روی زندگی‌ات گذاشته",
            "این تجربه واقعاً ناراحت‌کننده‌ست",
            "این موضوع واقعاً مهمه",
            "این تجربه واقعاً دشوار بوده",
            "این اتفاق تأثیر عمیقی رویت گذاشته",
            "این تجربه واقعاً دردناک بوده",
        ]

        self.question_phrases = [
            "حالت چطوره؟",
            "الان چه احساسی داری؟",
            "حال روحی‌ت چطوره؟",
            "چه حسی داری الان؟",
            "با توجه به اتفاقی که گفتی، الان چه احساسی داری؟",
            "این اتفاق چه تأثیری رویت گذاشته؟",
            "چه حسی نسبت به این موضوع داری؟",
            "اخیراً اتفاق خاصی افتاده که بخوای درباره‌اش صحبت کنی؟",
            "در چند روز گذشته حادثه‌ای رخ داده که روی حالت تأثیر گذاشته باشه؟",
            "اخیراً اتفاقی افتاده که بخوای درباره‌اش صحبت کنیم؟",
            "اتفاقی افتاده که بخوای درباره‌اش صحبت کنیم؟",
            "اخیراً تجربه خاصی داشتی؟",
            "این اتفاق چه زمانی رخ داده؟ اخیراً بوده یا مدتیه؟",
            "این رویداد اخیراً اتفاق افتاده یا مدتیه؟",
            "این اتفاق چه زمانی برات رخ داده؟",
            "این رویداد اخیراً بوده یا مدتیه؟",
            "چه زمانی این اتفاق افتاده؟",
            "این رخداد چه زمانی برات اتفاق افتاده؟",
            "این اتفاق اخیراً بوده یا مدتیه؟",
            "این رویداد چه زمانی رخ داده؟",
            "این حادثه اخیراً اتفاق افتاده؟",
            "این اتفاق چه مدت پیش رخ داده؟",
            "این رویداد در چند روز گذشته بوده؟",
            "این رخداد چه زمانی برات اتفاق افتاده؟",
            "این اتفاق اخیراً بوده یا مدتیه؟",
            "این رویداد چه زمانی رخ داده؟",
            "این حادثه اخیراً اتفاق افتاده؟",
            "این اتفاق چه مدت پیش رخ داده؟",
            "این رویداد در چند روز گذشته بوده؟"
        ]

    def get_user_day_progress(self, user):
        """Get or create day progress for a specific user"""
        day_progress, created = UserDayProgress.objects.get_or_create(user=user)
        return day_progress.calculate_current_day()


    def get_day_allowed_exercises(day):
        """Get allowed exercise numbers for a given day."""
        base_exercises = [0]
        
        if day == 8:
            return None  # All exercises are allowed
        elif 1 <= day <= 7:
            # Cumulative: day 1 = [1,2,3], day 2 = [1,2,3,4,5,6], etc.
            end_exercise = day * 3
            return base_exercises + list(range(1, end_exercise + 1))
        else:
            # Default to first day's exercises
            return base_exercises + [1, 2, 3]

    def parse_exercise_number(self, exercise_num):
        """Parse exercise number to base number (e.g., '2a' -> 2, '0.1' -> 0)"""
        if '.' in exercise_num:
            return int(float(exercise_num))
        else:
            import re
            base_num = re.match(r'(\d+)', exercise_num)
            return int(base_num.group(1)) if base_num else 0

    def filter_exercises_by_day(self, available_exercises, user):
        """Filter exercises based on user's current day"""
        current_day = self.get_user_day_progress(user)
        allowed_exercises = self.get_day_allowed_exercises(current_day)

        if allowed_exercises is None:
            return available_exercises

        # Filter exercises based on day restrictions
        filtered_exercises = []
        for exercise in available_exercises:
            exercise_base_num = self.parse_exercise_number(exercise["Exercise Number"])
            if exercise_base_num in allowed_exercises or exercise_base_num == 0:  # 0.x exercises allowed on day 1
                filtered_exercises.append(exercise)

        return filtered_exercises

    def get_user_state(self, user):
        """Get or create state for a specific user"""
        if user.id not in self.user_states:
            # Get the next session ID for new users
            from api.models import Message
            latest_session = Message.objects.filter(user=user).aggregate(Max('session_id'))['session_id__max']
            new_session_id = (latest_session or 0) + 1

            self.user_states[user.id] = {
                'state': "GREETING_FORMALITY_NAME",
                'message_count': 0,
                'emotion': None,
                'response': None,
                'stage': user.stage,
                'exercises_done': set(),
                'current_day': self.get_user_day_progress(user),
                'current_session_id': new_session_id,
                'emotion_message_count': 0,
                'event_message_count': 0
            }
        else:
            self.user_states[user.id]['current_day'] = self.get_user_day_progress(user)
            # Ensure current_session_id exists for existing users
            if 'current_session_id' not in self.user_states[user.id]:
                from api.models import Message
                latest_session = Message.objects.filter(user=user).aggregate(Max('session_id'))['session_id__max']
                self.user_states[user.id]['current_session_id'] = (latest_session or 0) + 1

            # Ensure state-specific counters exist for existing users
            if 'emotion_message_count' not in self.user_states[user.id]:
                self.user_states[user.id]['emotion_message_count'] = 0
            if 'event_message_count' not in self.user_states[user.id]:
                self.user_states[user.id]['event_message_count'] = 0
        return self.user_states[user.id]

    def transition(self, new_state, user):
        user_state = self.get_user_state(user)
        print(f"Transitioning from {user_state['state']} to {new_state}")
        user_state['state'] = new_state

        # Reset state-specific message counters when transitioning
        if new_state == "EMOTION":
            user_state['emotion_message_count'] = 0
        elif new_state == "SUPER_STATE_EVENT":
            user_state['event_message_count'] = 0

    def _load_sat_knowledge(self):
        """Load SAT knowledge base for injection into prompts"""
        try:
            with open('api/bot/RAG/sat_knowledge_base.md', "r", encoding="utf-8") as file:
                return file.read()
        except FileNotFoundError:
            return "دانش پایه SAT در دسترس نیست."

    def ask_llm(self, prompt_file, message, user, transition_info=None):
        with open(f'api/bot/Prompts/{prompt_file}', "r", encoding="utf-8") as file:
            system_prompt = file.read()

        # Load SAT knowledge base
        sat_knowledge = self._load_sat_knowledge()

        user_state = self.get_user_state(user)
        memory_context = self.memory_manager.format_memory_for_prompt(
            user,
            session_id=user_state.get('current_session_id')
        )
        session_history = self.memory_manager.get_formatted_session_history(
            user,
            session_id=user_state.get('current_session_id')
        )

        full_context = f"""### خلاصه اطلاعات کاربر:\n{memory_context}\n\n### تاریخچه کامل مکالمه این جلسه:\n{session_history}"""

        # Inject SAT knowledge into the prompt
        sat_knowledge_section = f"\n\n### 📚 دانش پایه تکنیک دلبستگی به خود (SAT):\n{sat_knowledge}\n"

        if transition_info:
            transition_context = f"\n\n### 🔍 وضعیت انتقال فعلی:\n**نتیجه انتقال:\n{transition_info}**\n"
            system_prompt = system_prompt.format(memory=full_context, transition_awareness=transition_context)
        else:
            system_prompt = system_prompt.format(memory=full_context)

        # Add SAT knowledge before repetition prevention
        system_prompt += sat_knowledge_section

        repetition_context = self._get_repetition_prevention_context()
        system_prompt += f"\n\n### ⚠️ هشدار مهم - جلوگیری از تکرار در کل مکالمه:\n{repetition_context}"

        # print(f'system_prompt={system_prompt}')
        # print(f'user_prompt={message}')

        response = openai_req_generator(system_prompt=system_prompt, user_prompt=message, json_output=False, temperature=0.1)

        self._track_response_for_repetition(response)

        return response

    def _get_repetition_prevention_context(self):
        """Generate context about used phrases to prevent repetition"""
        context = "**قبل از پاسخ دادن، این عبارات قبلاً استفاده شده‌اند و نباید تکرار شوند:**\n\n"

        if self.repetition_prevention.used_empathy_phrases:
            context += "**عبارات همدردی استفاده شده:**\n"
            for phrase in list(self.repetition_prevention.used_empathy_phrases)[-5:]:  # Show last 5
                context += f"- {phrase}\n"
            context += "\n"

        if self.repetition_prevention.used_questions:
            context += "**سوالات استفاده شده:**\n"
            for phrase in list(self.repetition_prevention.used_questions)[-5:]:  # Show last 5
                context += f"- {phrase}\n"
            context += "\n"

        # Add overused words warning
        overused_words = self.repetition_prevention.get_overused_words(threshold=2)
        if overused_words:
            context += "**⚠️ کلمات استفاده شده بیش از حد (نباید تکرار شوند):**\n"
            for word, count in overused_words.items():
                context += f"- '{word}' ({count} بار استفاده شده)\n"
            context += "\n"

        context += "**نکات مهم برای صحبت دوستانه:**\n"
        context += "1. هرگز هیچ یک از عبارات بالا را تکرار نکنید\n"
        context += "2. همیشه از عبارات جدید و متنوع استفاده کنید\n"
        context += "3. از بانک عبارات متنوع موجود استفاده کنید\n"
        context += "4. اگر تمام عبارات استفاده شده‌اند، عبارات جدید بسازید\n"
        context += "5. از تکرار عبارات 'متاسفم که'، 'می‌تونم بگم'، 'می‌خوام بدونم' خودداری کنید\n"
        context += "6. همیشه ابتدا تاریخچه مکالمه را بررسی کنید تا از تکرار جلوگیری کنید\n"
        context += "7. اگر کاربر قبلاً اطلاعاتی داده، به آن اشاره کنید و دوباره نپرسید\n"
        context += "8. از سوالات زمان‌محور متنوع استفاده کنید (چه زمانی، اخیراً، مدتیه)\n"
        context += "9. طبیعی و دوستانه صحبت کنید، نه مثل ربات\n"
        context += "10. از ایموجی‌ها به ندرت و فقط در مواقع مناسب استفاده کنید\n"
        context += "11. از تکرار ایموجی 😊 در هر پیام خودداری کنید\n"
        context += "12. حداکثر یک ایموجی در هر پیام استفاده کنید\n"
        context += "13. از تکرار کلمات کلیدی مثل 'کمک'، 'متاسفم'، 'می‌تونم' خودداری کنید\n"
        context += "14. به جای 'کمکت کنم' از عبارات متنوع استفاده کنید\n"
        context += "15. به جای 'متاسفم' از عبارات همدردی متنوع استفاده کنید\n"
        context += "16. به جای 'می‌تونم' از ساختارهای جملات متنوع استفاده کنید\n"

        return context

    def _track_response_for_repetition(self, response):
        """Track the response to prevent future repetition"""
        if not response:
            return

        # Extract sentences and track them
        sentences = re.split(r'[.!?؟]', response)
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and len(sentence) > 10:  # Only track meaningful sentences
                # Determine category based on content
                if any(word in sentence for word in ['احساس', 'حس', 'ناراحت', 'خوشحال', 'متاسفم', 'درکت', 'فهمم']):
                    self.repetition_prevention.add_phrase(sentence, "empathy")
                elif '?' in sentence or '؟' in sentence:
                    self.repetition_prevention.add_phrase(sentence, "question")
                else:
                    self.repetition_prevention.add_phrase(sentence, "general")

    def customize_excercises(self, prompt_file, user, excercises):
        with open(f'api/bot/Prompts/{prompt_file}', "r", encoding="utf-8") as file:
            system_prompt = file.read()
            
        # Load SAT knowledge base
        sat_knowledge = self._load_sat_knowledge()
        
        memory_context = self.memory_manager.format_memory_for_prompt(user)
        
        # Get current day progress
        current_day = self.get_user_day_progress(user)
        
        # with open('debug.md', 'w', encoding="utf-8") as file:
        #     file.write(memory_context)
        if memory_context != "":
            system_prompt = system_prompt.format(
                memory=memory_context, 
                exc=excercises,
                current_day=current_day)
        else:
            system_prompt = system_prompt.format(
                memory="", 
                exc=excercises,
                current_day=current_day)

        # Inject SAT knowledge into the prompt
        sat_knowledge_section = f"\n\n### 📚 دانش پایه تکنیک دلبستگی به خود (SAT):\n{sat_knowledge}\n"
        system_prompt += sat_knowledge_section

        return openai_req_generator(system_prompt=system_prompt, user_prompt=None, json_output=False, temperature=0.1)

    def if_transition(self, user, data):
        # Get current session messages
        messages_obj = self.memory_manager.get_chat_history(user)
        chat_history = "\n".join([
            f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
            for msg in messages_obj
        ])

        # For emotion and event states, only consider current session history
        # For other states (like greeting), consider both current session and previous memory
        if data in ["emotion.md", "event.md"]:
            # Only use current session history for emotion and event states
            full_context = f"تاریخچه جلسه فعلی:\n{chat_history}"
        else:
            # For other states, consider both current session and previous memory
            user_memory = self.memory_manager.get_current_memory(user)
            full_context = ""
            if user_memory:
                full_context += f"اطلاعات قبلی کاربر از جلسات گذشته:\n{user_memory}\n\n"
            if chat_history:
                full_context += f"تاریخچه جلسه فعلی:\n{chat_history}"

        transit = if_data_sufficient_for_state_change(data, full_context)
        return transit

    def state_handler(self, message, user):
        user_state = self.get_user_state(user)
        exercise_number = None

        messages_obj = self.memory_manager.get_chat_history(user)
        chat_history = "\n".join([
            f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
            for msg in messages_obj
        ])

        if user_state['state'] == "EMOTION_DECIDER":
            emotion = self.openai_llm.emotion_retriever(user_message=message, chat_history=chat_history)
            self.set_emotion(emotion, user)
            if 'Positive' in user_state['emotion']:
                self.transition("ASK_EXERCISE", user)
            else:
                self.transition("SUPER_STATE_EVENT", user)

        if user_state['state'] == "GREETING_FORMALITY_NAME":
            # Check transition status for greeting
            transit = self.if_transition(user, "greeting.md")
            response = self.ask_llm(
                "greeting_formality_name.md", message, user, transit)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "EMOTION":
            # Check transition status for emotion
            transit = self.if_transition(user, "emotion.md")
            response = self.ask_llm("emotion.md", message, user, transit)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "SUPER_STATE_EVENT":
            # Check transition status for event
            transit = self.if_transition(user, "event.md")
            response = self.ask_llm("ask_all_event.md", message, user, transit)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "OPEN_ENDED_CONVERSATION":
            response = self.ask_llm("open_ended_conversation.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "ASK_EXERCISE":
            response = self.ask_llm("ask_exercise.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "EXERCISE_SUGGESTION":
            user_memory = self.memory_manager.get_current_memory(user)

            all_exercises = exercises
            day_filtered_exercises = self.filter_exercises_by_day(all_exercises, user)

            exercise_content, exercise_number = suggest_exercises(
                user_state['exercises_done'],
                user_memory,
                user_state['stage'],
                day_filtered_exercises
            )

            if not exercise_content:
                # Handle case where no more exercises are available for the day
                response = "به نظر می‌رسه تمام تمرین‌های امروز رو انجام دادی. فردا تمرین‌های جدیدی خواهیم داشت. کارِت عالی بود!"
                self.transition("THANKS", user)
                return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

            user_state['exercises_done'].add(exercise_number)
            response = self.customize_excercises(
                "suggestion.md", user, exercise_content)
            explainability = create_exercise_explanation(
                user_memory, exercise_content)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), explainability, exercise_number

        elif user_state['state'] == "EXERCISE_EXPLANATION":
            response = self.ask_llm("exercise_explanation.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "FEEDBACK":
            response = self.ask_llm("feedback.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "LIKE_ANOTHER_EXERCSISE":
            response = self.ask_llm("like_to_do_another_exc.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "THANKS":
            response = self.ask_llm("thanks.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        elif user_state['state'] == "END":
            response = self.ask_llm("end.md", message, user)
            return response, create_recommendations(response, self.memory_manager.get_current_memory(user)), None, None

        else:
            return "میتونی بیشتر توضیح بدی", [], None, None

    def execute_state(self, message, user):
        user_id = user.id

        # Check if user is currently being processed
        if self.message_buffer.is_user_processing(user_id):
            # Add message to buffer and return None to indicate processing is ongoing
            self.message_buffer.add_message(user_id, message)
            return None, None, None, None, None

        # Start processing for this user
        self.message_buffer.start_processing(user_id)

        try:
            # Get any buffered messages and concatenate with current message
            buffered_messages = self.message_buffer.get_buffered_messages(user_id)
            if buffered_messages:
                # Concatenate buffered messages with current message
                all_messages = buffered_messages + [message]
                final_message = self.message_buffer.concatenate_messages(all_messages)
                print(f"Processing concatenated message for user {user_id}: {final_message}")
            else:
                final_message = message

            user_state = self.get_user_state(user)
            print(f"You are in the {user_state['state']} state")

            # Reset repetition prevention for new sessions
            if user_state['state'] == "GREETING_FORMALITY_NAME" and user_state['message_count'] == 0:
                self.repetition_prevention = RepetitionPrevention()

            # update memory and increment message count with current session ID
            self.memory_manager.add_message(
                user=user,
                text=final_message,
                is_user=True,
                session_id=user_state['current_session_id'],
                state=user_state['state']
            )

            # Check if we need to update memory summary (only for current session)
            if user_state['message_count'] >= 3:
                self.memory_manager.update_memory(user)
                user_state['message_count'] = 0

            if user_state['state'] == "GREETING_FORMALITY_NAME":
                transit = self.if_transition(user, "greeting.md")
                print("transit", transit)
                if "بله" in transit:
                    self.transition("EMOTION", user)

            elif user_state['state'] == "EMOTION":
                # Always process emotion state in each session
                # Track messages in current state
                if 'emotion_message_count' not in user_state:
                    user_state['emotion_message_count'] = 0
                user_state['emotion_message_count'] += 1

                transit = self.if_transition(user, "emotion.md")
                print("transit", transit)
                print(f"Emotion messages in current state: {user_state['emotion_message_count']}")

                # Only transition if we have sufficient emotion data AND at least 2 messages in this state
                if "بله" in transit and user_state['emotion_message_count'] >= 2:
                    self.transition("EMOTION_DECIDER", user)

            elif user_state['state'] == "SUPER_STATE_EVENT":
                # Always process event state in each session
                # Track messages in current state
                if 'event_message_count' not in user_state:
                    user_state['event_message_count'] = 0
                user_state['event_message_count'] += 1

                transit = self.if_transition(user, "event.md")
                print("transit", transit)
                print(f"Event messages in current state: {user_state['event_message_count']}")

                # Only transition if we have sufficient event data AND at least 2 messages in this state
                if "بله" in transit and user_state['event_message_count'] >= 2:
                    # Check if user wants to explain more (نمیدونی چی شد میخوای تعریف کنم برات)
                    if "نمیدونی" in message or "تعریف کنم" in message:
                        self.transition("OPEN_ENDED_CONVERSATION", user)
                    else:
                        self.transition("ASK_EXERCISE", user)

            elif user_state['state'] == "OPEN_ENDED_CONVERSATION":
                # After open-ended conversation, transition to ASK_EXERCISE
                # Check if conversation has reached a natural end point
                if len(messages_obj) >= 4:  # After sufficient conversation
                    self.transition("ASK_EXERCISE", user)

            elif user_state['state'] == "ASK_EXERCISE":
                messages_obj = self.memory_manager.get_chat_history(user)
                chat_history = "\n".join([
                    f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
                    for msg in messages_obj
                ])
                response = self.openai_llm.response_retriever(user_message=message, chat_history=chat_history)
                self.set_response(response, user)
                if 'Yes' in user_state['response']:
                    self.transition("EXERCISE_SUGGESTION", user)
                else:
                    self.transition("THANKS", user)

            elif user_state['state'] == "EXERCISE_SUGGESTION":
                messages_obj = self.memory_manager.get_chat_history(user)
                chat_history = "\n".join([
                    f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
                    for msg in messages_obj
                ])
                response = self.openai_llm.response_retriever(user_message=message, chat_history=chat_history)
                self.set_response(response, user)
                if 'Yes' in user_state['response']:
                    self.transition("EXERCISE_EXPLANATION", user)
                else:
                    self.transition("LIKE_ANOTHER_EXERCSISE", user)

            elif user_state['state'] == "EXERCISE_EXPLANATION":
                messages_obj = self.memory_manager.get_chat_history(user)
                chat_history = "\n".join([
                    f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
                    for msg in messages_obj
                ])
                response = self.openai_llm.response_retriever(user_message=message, chat_history=chat_history)
                self.set_response(response, user)
                if 'Yes' in user_state['response']:
                    self.transition("FEEDBACK", user)
                else:
                    self.transition("LIKE_ANOTHER_EXERCSISE", user)

            elif user_state['state'] == "FEEDBACK":
                self.transition("LIKE_ANOTHER_EXERCSISE", user)

            elif user_state['state'] == "LIKE_ANOTHER_EXERCSISE":
                # Use response_retriever to intelligently detect user's intent for another exercise
                messages_obj = self.memory_manager.get_chat_history(user)
                chat_history = "\n".join([
                    f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
                    for msg in messages_obj
                ])
                response = self.openai_llm.response_retriever(user_message=message, chat_history=chat_history)
                self.set_response(response, user)
                if 'Yes' in user_state['response']:
                    self.transition("EXERCISE_SUGGESTION", user)
                else:
                    self.transition("THANKS", user)

            elif user_state['state'] == "THANKS":
                self.transition("END", user)

            elif user_state['state'] == "END":
                print("State machine has reached the end.")

            response, recommendations, explainibility, excercise_number = self.state_handler(
                final_message, user)
            # print(response, recommendations)

            self.memory_manager.add_message(
                user=user,
                text=response,
                is_user=False,
                session_id=user_state['current_session_id'],
                state=user_state['state']
            )
            user_state['message_count'] += 1

            return response, recommendations, user_state['state'], explainibility, excercise_number

        finally:
            # Always end processing when done
            self.message_buffer.end_processing(user_id)

    def set_emotion(self, emotion, user):
        user_state = self.get_user_state(user)
        user_state['emotion'] = emotion
        print(f'user emotion={user_state["emotion"]}')

    def set_response(self, response, user):
        user_state = self.get_user_state(user)
        user_state['response'] = response
        print(f'user response={user_state["response"]}')

    def handle_session_end(self, user):
        """Handle cleanup when user ends session"""
        self.memory_manager.update_memory(user)
        # Reset user state
        self.user_states[user.id] = {
            'state': "GREETING_FORMALITY_NAME",
            'message_count': 0,
            'emotion': None,
            'response': None
        }

    def reset_state_machine(self, user):
        """Reset the state machine to initial state for a specific user"""
        # Get the next session ID
        from api.models import Message
        latest_session = Message.objects.filter(user=user).aggregate(Max('session_id'))['session_id__max']
        new_session_id = (latest_session or 0) + 1

        # Reset user state to initial values with new session
        self.user_states[user.id] = {
            'state': "GREETING_FORMALITY_NAME",
            'message_count': 0,
            'emotion': None,
            'response': None,
            'stage': user.stage,
            'exercises_done': set(),
            'current_day': self.get_user_day_progress(user),
            'current_session_id': new_session_id,
            'emotion_message_count': 0,
            'event_message_count': 0
        }

        # Reset repetition prevention for this user
        # self.repetition_prevention.reset_for_user(user.id)

        # Clear any buffered messages for this user
        self.message_buffer.end_processing(user.id)

        print(f"State machine reset for user {user.id} to initial state with session {new_session_id}")
        return {
            'state': "GREETING_FORMALITY_NAME",
            'message_count': 0,
            'emotion': None,
            'response': None,
            'stage': user.stage,
            'current_day': self.get_user_day_progress(user),
            'session_id': new_session_id
        }

    def process_buffered_messages(self, user):
        """Process any buffered messages for a user"""
        user_id = user.id

        # Check if user is currently being processed
        if self.message_buffer.is_user_processing(user_id):
            return None, None, None, None, None

        # Check if there are buffered messages
        if not self.message_buffer.has_buffered_messages(user_id):
            return None, None, None, None, None

        # Start processing for this user
        self.message_buffer.start_processing(user_id)

        try:
            # Get all buffered messages
            buffered_messages = self.message_buffer.get_buffered_messages(user_id)
            if buffered_messages:
                final_message = self.message_buffer.concatenate_messages(buffered_messages)
                print(f"Processing buffered messages for user {user_id}: {final_message}")

                # Process the concatenated message
                return self.execute_state(final_message, user)

            return None, None, None, None, None

        finally:
            # Always end processing when done
            self.message_buffer.end_processing(user_id)
