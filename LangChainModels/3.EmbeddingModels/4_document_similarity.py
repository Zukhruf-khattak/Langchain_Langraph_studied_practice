from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

load_dotenv()

embedding = OpenAIEmbeddings(model='text-embedding-3-large', dimensions=300)

documents = [
    "Babar Azam is a Pakistani cricketer known for his elegant batting and consistent performances.",
"Mohammad Rizwan is a Pakistani wicketkeeper-batter admired for his determination and reliable batting.",
"Shaheen Shah Afridi is a Pakistani fast bowler known for his pace, swing, and ability to take early wickets.",
"Fakhar Zaman is a Pakistani opening batter famous for his aggressive style and match-winning performances.",
"Shadab Khan is a Pakistani all-rounder known for his leg-spin bowling, useful batting, and athletic fielding."
]

query = 'tell me about babar azam'

doc_embeddings = embedding.embed_documents(documents)
query_embedding = embedding.embed_query(query)

scores = cosine_similarity([query_embedding], doc_embeddings)[0]

index, score = sorted(list(enumerate(scores)),key=lambda x:x[1])[-1]

print(query)
print(documents[index])
print("similarity score is:", score)