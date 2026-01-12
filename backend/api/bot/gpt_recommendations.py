

from api.bot.gpt import openai_req_generator


def create_recommendations(bot_message, memory):
    # Debug
    return ["", "", ""]
    suggestions = openai_req_generator(f"""You are tasked with assisting in a dialogue between two parties.
                          Given the message from one side, your goal is to generate three concise (each of them less than 3 words) and
                                        actionable suggestions that the other side
                          could provide in response. Each suggestion should be brief, contextually relevant, and formatted clearly as:
                          suggestion1 / suggestion2 / suggestion3.
                          Ensure that the responses are thoughtful, aligned with the conversation's context, and maintain a neutral tone.
                                       The writing style of suggestions should be informal.
                                       The output language should be in Farsi.
                                       Also you are given a memory of personal information of the user: {memory}.
                                       MESSAGE FROM ONE SIDE: {bot_message}.""", "")
    suggestions = suggestions.split("/")
    final_suggestions = []
    for suggestion in suggestions:
        final_suggestions.append(suggestion.replace(".", "").strip())
    return final_suggestions[0:3]