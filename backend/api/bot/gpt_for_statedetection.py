from api.bot.gpt import openai_req_generator

def if_data_sufficient_for_state_change(prompt_path, responses):
    with open("debug2.md", "w", encoding="utf-8") as file:
        file.write(f"Responses: {responses}")
    
    with open("api/bot/Information_prompts/Judge_prompt.md", "r", encoding="utf-8") as file:
        judge_prompt = file.read()

    with open(f"api/bot/Information_prompts/{prompt_path}", "r", encoding="utf-8") as file:
        prompt = file.read()
    
    return openai_req_generator(judge_prompt
                         + "\n---------------------\n"
                         + f"اطلاعات استخراج شده مورد نیاز:\n{prompt}\n"
                         + "-------------------\n"
                         + f"تاریخچه گفتگو:\n{responses}", "")
                        
