Your task is to identify the user's response. There are two classes (Yes and No) and you should generate only one token stating the response of the user's message.

## Examples of Yes responses (affirmative):
- "بله" (Yes)
- "آره" (Yeah)
- "بگو" (Say it)
- "می‌خوام" (I want it)
- "باشه" (Okay)
- "درسته" (Right)
- "مطمئنا" (Certainly)
- "حتما" (Definitely)
- "بله لطفا" (Yes please)
- "آره بگو" (Yeah say it)
- "می‌خوام ببینم" (I want to see)
- "باشه بگو" (Okay say it)
- "بله می‌خوام" (Yes I want)
- "آره می‌خوام" (Yeah I want)
- "بله لطفا بگو" (Yes please say it)
- "می‌خوام تمرین" (I want exercise)
- "بله تمرین" (Yes exercise)
- "آره تمرین" (Yeah exercise)
- "باشه تمرین" (Okay exercise)
- "بله بگو" (Yes say it)
- "آره بگو" (Yeah say it)
- "می‌خوام بگو" (I want say it)
- "باشه بگو" (Okay say it)
- "بله می‌خوام بگو" (Yes I want say it)
- "آره می‌خوام بگو" (Yeah I want say it)
- "باشه می‌خوام" (Okay I want)
- "بله باشه" (Yes okay)
- "آره باشه" (Yeah okay)
- "می‌خوام باشه" (I want okay)
- "بله درسته" (Yes right)
- "آره درسته" (Yeah right)
- "باشه درسته" (Okay right)
- "می‌خوام درسته" (I want right)
- "بله مطمئنا" (Yes certainly)
- "آره مطمئنا" (Yeah certainly)
- "باشه مطمئنا" (Okay certainly)
- "می‌خوام مطمئنا" (I want certainly)
- "بله حتما" (Yes definitely)
- "آره حتما" (Yeah definitely)
- "باشه حتما" (Okay definitely)
- "می‌خوام حتما" (I want definitely)

## Examples of No responses (negative):
- "نه" (No)
- "نمی‌خوام" (I don't want)
- "نه ممنون" (No thanks)
- "نه الان نه" (No not now)
- "نمی‌خوام الان" (I don't want now)
- "نه باشه" (No okay)
- "نمی‌خوام باشه" (I don't want okay)
- "نه درسته" (No right)
- "نمی‌خوام درسته" (I don't want right)
- "نه مطمئنا" (No certainly)
- "نمی‌خوام مطمئنا" (I don't want certainly)
- "نه حتما" (No definitely)
- "نمی‌خوام حتما" (I don't want definitely)

Example 1:
User Prompt: بگو
Your Response: Yes

Example 2:
User Prompt: نمی‌خوام
Your Response: No

Example 3:
User Prompt: می‌خوام تمرین
Your Response: Yes

Example 4:
User Prompt: نه ممنون
Your Response: No

WARNING 1: The user message's language is in Farsi.
WARNING 2: Only and Only generate one token from the list below:
WARNING 3: You should consider chat history for your judgement.
WARNING 4: If the user shows any form of interest, willingness, or positive response (even if they say "بگو" meaning "say it"), classify it as "Yes".
WARNING 5: If the user shows disinterest, refusal, or negative response, classify it as "No".

['Yes', 'No']