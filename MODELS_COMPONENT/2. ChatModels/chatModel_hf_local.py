from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
import os

# for installing huggingface models locally, we need to set the cache directory for huggingface models. By default, it is set to C:/Users/user/.cache/huggingface. We can change it to any directory we want. In this case, I have set it to D:/huggingface_cache.
os.environ['HF_HOME'] = 'D:/huggingface_cache' 


llm = HuggingFacePipeline.from_model_id(
    model_id='TinyLlama/TinyLlama-1.1B-Chat-v1.0',
    task='text-generation',
    pipeline_kwargs=dict(
        temperature=0.5,
        max_new_tokens=100
    )
)
model = ChatHuggingFace(llm=llm)

result = model.invoke("What is the capital of India")

print(result.content)