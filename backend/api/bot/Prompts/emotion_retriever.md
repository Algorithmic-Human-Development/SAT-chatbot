Your task is to identify the user's emotion. There are three classes (Negative, Antisocial, and Positive) and you should generate only one token stating the emotion of the user's message.

Example 1:
User Prompt: امروز خیلی حال خوبی دارم.
Your Response: Positive

Example 2:
User Prompt: امروز حالم گرفته است و احساس غم در وجودم حس می‌کنم.
Your Response: Negative


WARNING 1: The user message's language is in Farsi.
WARNING 2: Only and Only generate one token from the list below:
WARNING 3: You should consider chat history for your judgement.

['Negative', 'Antisocial', 'Positive']