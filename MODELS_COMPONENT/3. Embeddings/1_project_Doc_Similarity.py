from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# cosine_similarity is a measure of similarity between two non-zero vectors in an inner product space. It is defined as the cosine of the angle between them, which ranges from -1 to 1. A value of 1 indicates that the vectors are identical, while a value of -1 indicates that they are completely opposite. A value of 0 indicates that the vectors are orthogonal (i.e., they have no similarity). In the context of document similarity, cosine similarity can be used to compare the embeddings of documents and queries to determine how closely they match.
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Virat Kohli is an Indian cricketer known for his aggressive batting and leadership.",
    "MS Dhoni is a former Indian captain famous for his calm demeanor and finishing skills.",
    "Sachin Tendulkar, also known as the 'God of Cricket', holds many batting records.",
    "Rohit Sharma is known for his elegant batting and record-breaking double centuries.",
    "Jasprit Bumrah is an Indian fast bowler known for his unorthodox action and yorkers."
]

query = 'tell me about bumrah'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0] # cosine_similarity returns a 2D array, we take the first row to get the similarity scores for the query against each document.

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1] # enumerate(scores) gives us pairs of (index, score), we sort them by score and take the last one which has the highest similarity score.

print(query)
print(documents[index])
print("similarity score is:", score)



