from superjsonmode.integrations.openai import StructuredOpenAIModel
from openai import OpenAI
from pydantic import BaseModel
import time
import json

class color:
   CYAN = '\033[96m'
   BOLD = '\033[1m'
   END = '\033[0m'


print("\n" + color.BOLD + "Generating JSON naively with OpenAI gpt-3.5-turbo..." + color.END)
print("-------------------------------------------")

prompt = """Luke Skywalker is a famous character."""

start = time.time()

default_prompt = f"""{prompt}

Based on the prompt above, generate a JSON blob with the following keys: "name", "genre", "age", "race", "occupation", "best_friend", and "home_planet".
"""

print(default_prompt)

client = OpenAI()

completion = client.chat.completions.create(
  model="gpt-3.5-turbo",
  messages=[
    {"role": "user", "content": default_prompt}
  ]
)

print("Total time: " + color.CYAN + color.BOLD + f"{time.time() - start}" + color.END)
print("-------------------------------------------")
print(completion.choices[0].message.content)
print("-------------------------------------------\n")

print("\n" + color.BOLD + "Testing same OpenAI model with Super JSON Mode..." + color.END)
print("-------------------------------------------")
model = StructuredOpenAIModel()

class Character(BaseModel):
    name: str
    genre: str
    age: int
    race: str
    occupation: str
    best_friend: str
    home_planet: str

prompt_template = """{prompt}

Please fill in the following information about this character for this key. Keep it succinct. It should be a {type}.

{key}: """

start = time.time()
output = model.generate(
    prompt,
    extraction_prompt_template=prompt_template,
    schema=Character,
    batch_size=7,
    # stop=["\n\n"],
    temperature=0,
)
print("Total time: " + color.CYAN + color.BOLD + f"{time.time() - start}" + color.END)
print("-------------------------------------------")
# Total Time: 0.409s

print(json.dumps(output, indent=2))
# {
#     "name": "Luke Skywalker",
#     "genre": "Science fiction",
#     "age": "23",
#     "race": "Human",
#     "occupation": "Jedi Knight",
#     "best_friend": "Han Solo",
#     "home_planet": "Tatooine",
# }
print("-------------------------------------------")


