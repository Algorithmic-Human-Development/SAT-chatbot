from api.models import Message, UserMemoryState
from api.bot.gpt_for_summarization import openai_req_generator
from django.db.models import Max
import time


def summarize_conversation(text, context):
    system_prompt = f"""
    با توجه به پیش‌زمینه‌ای که از مکالمه موجود است و به شرح زیر است:
    {context}،
    اطلاعات خلاصه را بروز کنید و این اطلاعات جدید که در زیر آمده است را به خلاصه‌ی پیشین اضافه کنید:
    {text}

    در خلاصه‌سازی نهایی خود ، تلاش کنید که فقط صحبت‌های کاربر را خلاصه کنید و بر اساس اطلاعات جدید و خلاصه‌ی پیشین، به احساسات کاربر، اطلاعات شخصی کاربر، حالات روحی کاربر، وقایعی که اخیرا برای او رخ داده است و غیره، توجه ویژه داشته باشید.
    همچنین مواردی همچون نام کاربر، ترجیح کاربر برای صحبت کردن به حالت رسمی یا دوستانه و حال کاربر را به وضوح با تیتر مشخص ذخیره کنید.
    """
    return openai_req_generator(system_prompt=system_prompt, json_output=False, temperature=0.1)


class MemoryManager:
    def __init__(self):
        pass

    def get_or_create_memory_state(self, user):
        print(user, type(user))
        memory_state, created = UserMemoryState.objects.get_or_create(user=user)
        return memory_state

    def add_message(self, user, text, is_user=True, session_id=None, state=None):
        # Get the current session ID or create new one
        if session_id is None:
            current_session = Message.objects.filter(user=user).aggregate(Max('session_id'))['session_id__max']
            session_id = (current_session or 0) + 1

        # Create and save the message
        message = Message.objects.create(
            user=user,
            text=text,
            session_id=session_id,
            is_user=is_user,
            state=state,
            timestamp=time.time()
        )
        return message

    def get_unprocessed_messages(self, user, session_id=None):
        memory_state = self.get_or_create_memory_state(user)
        last_processed = memory_state.last_processed_message

        query = Message.objects.filter(user=user)

        # Filter by session if provided
        if session_id is not None:
            query = query.filter(session_id=session_id)

        if last_processed:
            return query.filter(timestamp__gt=last_processed.timestamp).order_by('timestamp')
        return query.order_by('timestamp')

    def get_session_messages(self, user, session_id):
        """Get all messages for a specific session"""
        return Message.objects.filter(user=user, session_id=session_id).order_by('timestamp')

    def get_current_session_messages(self, user):
        """Get messages from the current session"""
        current_session = Message.objects.filter(user=user).aggregate(Max('session_id'))['session_id__max']
        if current_session:
            return self.get_session_messages(user, current_session)
        return Message.objects.none()

    def update_memory(self, user):
        memory_state = self.get_or_create_memory_state(user)
        unprocessed_messages = self.get_unprocessed_messages(user)

        if not unprocessed_messages.exists():
            return memory_state.current_memory

        # Format messages for summarization
        conversation = "\n".join([
            f"{'User' if msg.is_user else 'LLM'}: {msg.text}"
            for msg in unprocessed_messages
        ])

        # Update the memory using LLM
        updated_memory = summarize_conversation(
            text=conversation,
            context=memory_state.current_memory
        )

        # Update the memory state
        memory_state.current_memory = updated_memory
        memory_state.last_processed_message = unprocessed_messages.last()
        memory_state.save()

        return updated_memory

    def get_chat_history(self, user, session_id=None):
        query = Message.objects.filter(user=user)
        if session_id is not None:
            query = query.filter(session_id=session_id)
        return query.order_by('timestamp')

    def get_current_memory(self, user):
        memory_state = self.get_or_create_memory_state(user)
        return memory_state.current_memory

    def end_session(self, user):
        self.update_memory(user)

    def get_formatted_session_history(self, user, session_id=None):
        """
        Returns the full conversation history for the current session,
        formatted as alternating User/Assistant messages.
        """
        messages = self.get_chat_history(user, session_id)
        lines = []
        for msg in messages:
            role = "User" if msg.is_user else "Assistant"
            lines.append(f"{role}: {msg.text}")
        return "\n".join(lines)

    def format_memory_for_prompt(self, user, session_id=None):
        # Get current memory state
        memory_state = self.get_or_create_memory_state(user)
        current_memory = memory_state.current_memory or ""

        # Get unprocessed messages for current session
        unprocessed = self.get_unprocessed_messages(user, session_id)
        unprocessed_text = "\n".join([
            f"{'User' if msg.is_user else 'Assistant'}: {msg.text}"
            for msg in unprocessed
        ])

        # Combine current memory with unprocessed messages
        if unprocessed_text:
            formatted_memory = f"""پیش‌زمینه مکالمه:\n{current_memory}\n\nپیام‌های اخیر:\n{unprocessed_text}"""
        else:
            formatted_memory = f"""پیش‌زمینه مکالمه:\n{current_memory}"""
        return formatted_memory.strip()
