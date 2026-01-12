from api.bot.gpt import openai_req_generator

def create_exercise_explanation(memory: str, exercise_content: str) -> str:
    """
    Creates personalized explanation for why a specific exercise is recommended based on user's memory
    
    Args:
        memory: String containing user's personal information and context
        exercise_content: Exercise content to explain
        
    Returns:
        Single string containing personalized explanation
    """
    try:
        with open('api/bot/RAG/exercise_explanation_prompt.md', 'r', encoding='utf-8') as f:
            prompt_template = f.read()
    except FileNotFoundError:
        return "خطا: قالب توضیحات یافت نشد"

    formatted_prompt = prompt_template.format(
        memory=memory,
        exercise_descriptions=exercise_content
    )

    return openai_req_generator(formatted_prompt, "").strip()