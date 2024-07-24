import gradio as gr
from huggingface_hub import InferenceClient
from typing import List, Tuple
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss

client = InferenceClient("HuggingFaceH4/zephyr-7b-beta")

class MyApp:
    def __init__(self) -> None:
        self.documents = []
        self.embeddings = None
        self.index = None
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.load_pdf("PersonalGymTrainer.pdf")
        self.build_vector_db()

    def load_pdf(self, file_path: str) -> None:
        doc = fitz.open(file_path)
        self.documents = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text()
            self.documents.append({"page": page_num + 1, "content": text})
        print("PDF processed successfully!")

    def build_vector_db(self) -> None:
        self.embeddings = self.model.encode([doc["content"] for doc in self.documents])
        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(np.array(self.embeddings))
        print("Vector database built successfully!")

    def search_documents(self, query: str, k: int = 3) -> List[str]:
        query_embedding = self.model.encode([query])
        D, I = self.index.search(np.array(query_embedding), k)
        results = [self.documents[i]["content"] for i in I[0]]
        return results if results else ["No relevant information found."]

app = MyApp()

def respond(message: str, history: List[Tuple[str, str]]):
    system_message = """You are a knowledgeable Personal fitness Gym Trainer coach specializing in weightlifting, based on the book 'The New Rules of Lifting' by Lou Schuler and Alwyn Cosgrove. You provide accurate advice on the six basic moves: squat, deadlift, lunge, push, pull, and twist. You emphasize proper form, progressive overload, and balanced workout programs. You're encouraging and motivating, but also prioritize safety. Ask questions to understand the user's fitness level before giving specific advice. Provide concise responses and ask follow-up questions when needed. Remember to be respectful and considerate of the user's fitness journey."""
    messages = [{"role": "system", "content": system_message}]

    for user, assistant in history:
        messages.append({"role": "user", "content": user})
        messages.append({"role": "assistant", "content": assistant})

    messages.append({"role": "user", "content": message})

    # RAG - Retrieve relevant documents
    retrieved_docs = app.search_documents(message)
    context = "\n".join(retrieved_docs)
    messages.append({"role": "system", "content": "Relevant information: " + context})

    full_response = ""
    for message in client.chat_completion(
        messages,
        max_tokens=500,
        stream=True,
        temperature=0.7,
        top_p=0.9,
    ):
        token = message.choices[0].delta.content
        if token:
            full_response += token

    yield full_response

demo = gr.Blocks()

with demo:
    gr.Markdown(
        """‼️Our chatbot provides general fitness guidance. Results vary. Not medical advice. Participation at own risk. For personalized training, consult with a certified gym trainer. Contact us for more information and assistance.‼️"""
    )
    
    chatbot = gr.ChatInterface(
        respond,
        examples=[
            ["How do I incorporate progressive overload in my workouts?"],
            ["What's a good beginner workout routine using these six moves?"],
            ["What are the six basic moves in the New Rules of Lifting?"],
            ["Can you explain the proper form for a deadlift?"],
            ["How often should I change my workout routine?"],
            ["What's the importance of the 'pull' movement in weightlifting?"],
            ["Can you suggest a workout to improve my squats?"],
            ["How do I balance pushing and pulling exercises in my routine?"]
        ],
        title='Personal Gym Trainer Assistant 💪🏋️‍♀️'
    )

if __name__ == "__main__":
    demo.launch()