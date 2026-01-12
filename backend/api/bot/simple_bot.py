import os
import random
import re
import json
from api.bot.gpt import openai_req_with_history
from api.bot.gpt_recommendations import create_recommendations
from api.models import UserDayProgress

PROMPT_PATH = 'api/bot/Prompts/simple_fsm_full.md'
EXERCISES_DIR = 'api/bot/RAG/Exercises'

# Load exercises metadata from JSON file, similar to llm_excercise_suggestor.py
with open('api/bot/RAG/exercises_mapping.json', 'r', encoding='utf-8') as f:
    exercises_metadata = json.load(f)


def load_sat_knowledge():
    """Load SAT knowledge base for injection into prompts"""
    try:
        with open('api/bot/RAG/sat_knowledge_base.md', "r", encoding="utf-8") as file:
            return file.read()
    except FileNotFoundError:
        return "دانش پایه SAT در دسترس نیست."

def load_system_prompt():
    with open(PROMPT_PATH, 'r', encoding='utf-8') as f:
        return f.read()


def get_user_day_progress(user):
    """Get or create day progress for a specific user."""
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


def parse_exercise_number(exercise_num):
    """Parse exercise number to base number (e.g., '2a' -> 2, '0.1' -> 0)."""
    if '.' in exercise_num:
        return int(float(exercise_num))
    else:
        base_num = re.match(r'(\d+)', exercise_num)
        return int(base_num.group(1)) if base_num else 0


def get_daily_exercises(user, count=3):
    """Get daily exercises from the exercises directory based on user's daily progress."""
    current_day = get_user_day_progress(user)
    allowed_exercise_nums = get_day_allowed_exercises(current_day)

    # Filter the full list of exercises based on the allowed numbers for the day
    if allowed_exercise_nums is not None:
        available_exercises = [
            exercise for exercise in exercises_metadata
            if parse_exercise_number(exercise["Exercise Number"]) in allowed_exercise_nums
        ]
    else:
        # If allowed_exercise_nums is None (day 8+), all exercises are available
        available_exercises = exercises_metadata

    if not available_exercises:
        return "به نظر می‌رسه تمام تمرین‌های امروز رو انجام دادی. فردا تمرین‌های جدیدی خواهیم داشت. کارِت عالی بود!"

    # Get daily exercise numbers from the filtered list
    selected_exercises = random.sample(available_exercises, min(count, len(available_exercises)))
    selected_exercise_numbers = [ex["Exercise Number"] for ex in selected_exercises]

    print(f"selected_exercise_numbers: {selected_exercise_numbers}")

    # Fetch the content of the selected exercises
    exercises_content = []
    for exercise_num in selected_exercise_numbers:
        file_path = os.path.join(EXERCISES_DIR, f'exercise{exercise_num}.txt')
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read().strip()
                exercises_content.append(f"تمرین {exercise_num}: {content}")
        except Exception as e:
            print(f"Error reading {file_path}: {e}")
            continue

    return "\n\n".join(exercises_content)


def simple_bot_response(history, user_message, user):
    """
    Accepts a list of previous messages (history), a new user message, and the user object.
    Returns: (response, recommendations, updated_history)
    """
    daily_exercises = get_daily_exercises(user, 3)
    
    # Get current day progress
    current_day = get_user_day_progress(user)

    print(f"History: {history}")
    print(f"User is on Day {current_day}")

    # Load SAT knowledge base
    sat_knowledge = load_sat_knowledge()

    system_prompt = load_system_prompt()
    formatted_system_prompt = system_prompt.format(
        daily_exercises=daily_exercises, 
        memory="",
        current_day=current_day
    )

    # Inject SAT knowledge into the prompt
    sat_knowledge_section = f"\n\n### 📚 دانش پایه تکنیک دلبستگی به خود (SAT):\n{sat_knowledge}\n"
    formatted_system_prompt += sat_knowledge_section

    messages = [{"role": "system", "content": formatted_system_prompt}]
    # Include only the last six messages from history (if any)
    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})

    # print("messages:", messages)

    response = openai_req_with_history(messages, temperature=0.4)
    updated_history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]
    recommendations = create_recommendations(response, memory="")
    return response, recommendations, updated_history
