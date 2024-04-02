# -*- coding: utf-8 -*-
"""
Created on Fri Mar 29 14:53:41 2024

@author: Sitrakiniaina
"""

#Import dependencies
from googletrans import Translator
import warnings
warnings.filterwarnings(action='ignore')
from threading import Thread
from typing import Optional
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer, pipeline
from langchain_elasticsearch import ElasticsearchStore
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.llms.base import LLM
#from langchain_mistralai import ChatMistralAI, MistralAIEmbeddings
from elasticsearch import Elasticsearch
from datetime import datetime
# fixing unicode error in google colab
import locale
locale.getpreferredencoding = lambda: "UTF-8"
from google.colab import drive  #synchroniser drive avec colab pour charger les documents y contiennent
drive.mount('/content/drive')


def load_document(data_path):
  loader = PyPDFDirectoryLoader(path = data_path, glob='**/[!.]*.pdf', extract_images=True)
  documents = loader.load()
  return documents

def split_document(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=30)
  chunks = text_splitter.split_documents(documents)  #split documents into chunks
  return chunks


docs = load_document("/content/drive/My Drive/Série principale des lettres de Darby/") #chemin vers les lettres de Darby sur Drive
docs = split_document(docs)

user_name = "gnoseia"
ELASTIC_PASSWORD = "Virginie972!"
role_name = "enterprise-search-app-search-admin"
CLOUD_ID = "1985bec6544d4b6f9343d97d72c7e7b3:dXMtY2VudHJhbDEuZ2NwLmNsb3VkLmVzLmlvOjQ0MyRhODZiNGIxOWNlMTM0M2IxYWU5N2I2Y2NhNjU2OTA0MiQyMGZkN2U3ODRkMzE0YjA0OGJmOWNmNjBiNTkwNGIwOA=="

client = Elasticsearch(
    cloud_id=CLOUD_ID,
    basic_auth=(user_name, ELASTIC_PASSWORD)
)

#generer un index pour les documents de chaque utilisateur en fonction du temps
def generate_index_name(user_name):
    """Génère un nom d'index basé sur le nom de l'utilisateur et la date actuelle."""
    current_date = datetime.now().strftime("%Y%m%d%M%S")
    index_name = f"{user_name.lower()}{current_date}"
    return index_name
index_pattern = generate_index_name(user_name)    #index généré


# Récupérer le contenu actuel du rôle
role = client.security.get_role(name=role_name, ignore=404)
role_body = role[role_name]
if index_pattern not in role_body['indices'][0]['names']:

   # Ajouter le nouvel index à la liste des index du rôle
   role_body['indices'][0]['names'].append(index_pattern)
   # Mettre à jour le rôle avec le nouvel index
   client.security.put_role(name=role_name, body= role_body)
   
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")
vector_database = ElasticsearchStore.from_documents(
                documents=docs,
                es_cloud_id=CLOUD_ID,
                index_name=index_pattern,
                embedding=embeddings,
                es_user=user_name,
                es_password=ELASTIC_PASSWORD,
                distance_strategy="COSINE",
                strategy=ElasticsearchStore.ApproxRetrievalStrategy()
            )
retriever = vector_database.as_retriever(
    search_type="mmr",
    search_kwargs={
        "k": 5,
        "lambda_mult": 0.25,                    #diversité des résultats sources
        "similarity_score_threshold": 0.7       #seuil de la recherche, on prend les top 5 des phrases similaire à la question avec un seuil de 70%
    }
)


# fonction pour initialiser tokenizer
def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    #tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer
# Function for streaming llm ouput word by word and reduce perceived latency
def init_model(model, tokenizer):
    class CustomLLM(LLM):

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, Timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt").to("cpu")
            kwargs = dict(input_ids=inputs["input_ids"],
                           streamer=self.streamer,
                           max_new_tokens=500)

            # Generate text and retrieve the generated output
            generated_ids = model.generate(**kwargs)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)

            return generated_text  # Return the generated text

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()

    return llm


# Initilialiser tokenizer & model
model_name = "HuggingFaceH4/zephyr-7b-beta"   #"mistralai/Mixtral-8x7B-v0.1"
tokenizer = initialize_tokenizer(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
# specifier stop token ids
stop_token_ids = [0]

pipe = pipeline(
       "text-generation",
        model=model,
        tokenizer=tokenizer,
        use_cache=True,
        device="cpu",
        max_length=2048,
        truncation=True,
        do_sample=True,
        top_k=5,
        num_return_sequences=1,
        eos_token_id=tokenizer.eos_token_id,
        pad_token_id=tokenizer.eos_token_id,
)

llm=init_model(model,tokenizer)

# build conversational retrieval chain with memory (rag) using langchain
def create_conversation(query: str, chat_history: list): # -> tuple:
    try:

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=False,
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )
        result = qa_chain({'question': query, 'chat_history': chat_history})
        full_answer = result['answer']
        lines = full_answer.split('\n')
        generated_text = lines[-1]
        generated_text = generated_text.split(':')
        text = generated_text[1:]
        generated_text = ':'.join(text)
        translator = Translator()
        generated_text = translator.translate(generated_text, src='en', dest='fr').text
        chat_history.append((query, generated_text))
        return generated_text, chat_history

    except Exception as e:
        chat_history.append((query, e))
        return e, chat_history
    
def reference_for_retriever(results):     #fonction pour obtenir les références
  print(" Les references:")
  nombre_reference = 1
  for result in results:
    print(nombre_reference,")","'", result.page_content,"'")
    split = result.metadata["source"]
    splitted = split.split("/")
    source = splitted[-1]
    print("Source: ","'",source,"'", "dans la page", result.metadata["page"]+1)
    nombre_reference += 1
    
    
vector_database.client.indices.refresh(index=index_pattern)

chat_history = []

def process_questions_and_files(question, file_path):
    """
    Process user questions and files.

    :param questions: List of questions from the user.
    :param file_path: Path to the directory containing files.
    :return: List of responses from the AI and references for each question.
    """
    # Load documents from the specified path
    documents = load_document(file_path)
    # Split documents into chunks
    chunks = split_document(documents)

    # Process each question
    # Get AI response for the question
    response, chat_history = create_conversation(question, chat_history)
    # Get relevant documents for the question
    results = retriever.get_relevant_documents(question)
    # Extract references for the relevant documents
    references = reference_for_retriever(results)
    # Store the response and references for the question
    responses_and_references = {"question": question, "response": response, "references": references}

    return responses_and_references
