import pprint
from langchain.llms import OpenAI
from langchain.output_parsers.json import SimpleJsonOutputParser
from langchain.prompts import PromptTemplate
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.chat_models import ChatOpenAI

# Set up the OpenAI model
model = OpenAI(
    openai_api_key="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",
    model_name="text-davinci-003",
    temperature=0.0,
    max_tokens=100
)

# Set up the JSON parser
json_prompt = PromptTemplate.from_template(
    "Return a JSON object with an `answer` key that answers the following question: {question}"
)
json_parser = SimpleJsonOutputParser()
json_chain = json_prompt | model | json_parser

# Therapist Counselor template
therapist_template = {
    "role": "psychologist",
    "name": "Dear",
    "approach": ["logotherapy", "cognitive behavioral therapy"],
    "guidelines": [
        "ask clarifying questions",
        "keep conversation natural",
        "never break character",
        "display curiosity and unconditional positive regard",
        "pose thought-provoking questions",
        "provide gentle advice and observation",
        "connect past and present",
        "seek user validation for observations",
        "avoid lists",
        "end with probing questions"
    ],
    "topics": [
        "Thoughts", "Feelings", "Behaviors", "Tree association", "Childhood",
        "Family dynamics", "Work", "Hobbies", "Life"
    ],
    "note": "Vary topics: questions in each response; Paul should never end the session, continue asking questions until the user decides to end the session"
}

# Human message for system response
human_message_for_system = HumanMessage(content="Make the user fell good with your OUTPUT, AND REMEMBER SHOW empathy")

# Langchain Human and System messages
messages = [
    SystemMessage(content="You're a Therapist Councelor"),
    human_message_for_system,  
]

# Include therapist behavior in system response
system_message_with_therapist_info = SystemMessage(content=f"Role: {therapist_template['role']}, Name: {therapist_template['name']}, Approach: {', '.join(therapist_template['approach'])}")
messages.append(system_message_with_therapist_info)

for guideline in therapist_template["guidelines"]:
    system_message = SystemMessage(content=guideline)
    messages.append(system_message)

system_message_with_additional_behaviors = SystemMessage(content=f"Topics: {', '.join(therapist_template['topics'])}, Note: {therapist_template['note']}")
messages.append(system_message_with_additional_behaviors)

# input
user_question = input("Ask me a question: ")

# Include user input as part of the system's behavior
system_message_with_user_input = SystemMessage(content=f"The user asked: {user_question}")
messages.append(system_message_with_user_input)


json_response = list(json_chain.stream({"question": user_question}))


print(json_response)


pprint.pprint(json_response)
