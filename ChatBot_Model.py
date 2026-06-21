from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline
from langchain_core.messages import SystemMessage,HumanMessage,AIMessage


llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        task="text-generation",
                                        pipeline_kwargs={"temperature":0.9,"max_new_tokens": 256,
                                                         "do_sample":"False","repetition_penalty":1.03})


model = ChatHuggingFace(llm =llm)

print("________Choose Your AI Model Mode_________")
print("Press 1 for Happy")
print("Press 2 for Sad")
print("Press 3 for Angry")
print("Press 4 for Sarcastic")
print("Press 5 for Funny")
print("press 6 for Mature")

choice = input("Enter your choice: ")
if choice == "1":
    modee = "You are an angry AI agent. You respond aggressively and impatiently."
elif choice == "2":
    modee = "You are a very funny AI agent. You respond with humor and jokes."
elif choice == "3":
    modee = "You are a very sad AI agent. You respond in a depressed and emotional tone."
elif choice == "4":
    modee = "You are a sarcastic AI agent. You respond with irony and cynicism."
elif choice == "5":
    modee = "You are a humorous AI agent. You respond with jokes and comedic timing."
elif choice == "6":
    modee = "You are a mature AI agent. You respond with wisdom and depth."  


messages = [
    SystemMessage(content=modee)
]
print("________Enter 0 to exit the application_________")
while True:
    prompt = input("You : ")
    messages.append(HumanMessage(content=prompt))
    if prompt == "0":
        break

    response = model.invoke(prompt)
    messages.append(AIMessage(content = response.content))
    print("Bot : ",response.content)

print(messages)