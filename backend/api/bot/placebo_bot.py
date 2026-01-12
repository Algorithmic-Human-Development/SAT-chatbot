from api.bot.gpt import openai_req_with_history
from api.bot.gpt_recommendations import create_recommendations

PLACEBO_SYSTEM_PROMPT = """تو دستیار روان‌درمانی به خود هستی که وظیفه‌ات کمک به بهتر شدن حال روحی کاربر است."""


def placebo_bot_response(history, user_message, user):
    """
    Placebo bot with minimal functionality - just a simple chatbot with basic prompt.
    Accepts a list of previous messages (history), a new user message, and the user object.
    Returns: (response, recommendations, updated_history)

    Note: user parameter is kept for API consistency but not used in placebo version
    """
    print(f"Placebo Bot History: {history}")

    messages = [{"role": "system", "content": PLACEBO_SYSTEM_PROMPT}]

    if history:
        messages.extend(history[-6:])
    messages.append({"role": "user", "content": user_message})

    response = openai_req_with_history(messages, temperature=0.4)

    updated_history = (history or []) + [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": response}
    ]

    recommendations = create_recommendations(response, memory="")

    return response, recommendations, updated_history
