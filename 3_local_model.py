from langchain_huggingface import ChatHuggingFace,HuggingFacePipeline

llm = HuggingFacePipeline.from_model_id(model_id="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                                        task="text-generation",
                                        pipeline_kwargs={"max_new_tokens": 128,"do_sample":"False","repetition_penalty":1.03})
model = ChatHuggingFace(llm = llm)

response = model.invoke("legacy of virat kohli in test cricket?")   
print(response.content) 
        