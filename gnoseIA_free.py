# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 17:03:38 2024

@author: Boniface
"""

# !pip install  langchain
# !pip install  langchain_community
# !pip install  pypdf -U
# !pip install  transformers==4.38.2
# !pip install  tokenizers==0.15.2
# !pip install  chromadb
# !pip install  langchain_chroma
# !pip install  rapidocr-onnxruntime
# !pip install  langchain_cohere
# !pip install  cohere
# !pip install  sentence-transformers


#Import dependencies
import re
import inspect
import os
os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
import chromadb
from chromadb.config import Settings
import uuid
from cohere import Client
import warnings
warnings.filterwarnings(action='ignore')
from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer
from threading import Thread
from typing import Optional
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, LLMChain
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.llms import Cohere
from langchain.llms.base import LLM
from langchain.retrievers.contextual_compression import ContextualCompressionRetriever
from langchain_cohere import CohereRerank
from datetime import datetime



def load_document(data_path):
  loader = PyPDFDirectoryLoader(path = data_path, glob='**/[!.]*.pdf', extract_images=True)
  documents = loader.load()
  return documents

def split_document(documents):
  text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
  chunks = text_splitter.split_documents(documents)   #split documents into chunks
  return chunks


docs = load_document("https://gnoseia-corpus-storage.s3.eu-west-3.amazonaws.com/Corpus+Legislative/Front_populaire/")
docs_2 = load_document("https://gnoseia-corpus-storage.s3.eu-west-3.amazonaws.com/Corpus+Legislative/UDD/")
docs_3 = load_document("https://gnoseia-corpus-storage.s3.eu-west-3.amazonaws.com/Corpus+Legislative/Renaissance/")
docs_gnose = load_document("https://gnoseia-corpus-storage.s3.eu-west-3.amazonaws.com/Corpus+gnoseia/")


docs = split_document(docs)
docs_2 = split_document(docs_2)
docs_3 = split_document(docs_3)
docs_gnose = split_document(docs_gnose)


def init_chroma(data:list, entier:int):
  current_date = datetime.now().strftime("%Y%m%d%H%M%S")
  collection_name = f"chromadb_{current_date}{entier}"
  try:
    chroma_client = chromadb.Client()
    collection = chroma_client.create_collection(
       name=collection_name,
       metadata={"hnsw:space": "cosine"}
       )
    for doc in data:
      collection.add(
         ids=[str(uuid.uuid1())], metadatas=doc.metadata, documents=doc.page_content
      )
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    database = Chroma(
       client = chroma_client,
       collection_name = collection_name,
       embedding_function=embeddings
       )
    return database, collection
  except Exception as e:
    print(f"An error occured: {e}")
    return None, None

vector_database, collection = init_chroma(docs, 0)
vector_database_2, collection_2 = init_chroma(docs_2, 1)
vector_database_3, collection_3 = init_chroma(docs_3, 2)
vector_database_gnose, collection_gnose = init_chroma(docs_gnose, 3)

def retrieval(database):
  retriever = database.as_retriever(
     search_type="similarity_score_threshold",
     search_kwargs={
        "k": 15,                      #nombre des résultats tirés
        "score_threshold": 0.5        #seuil de la recherche, on prend les top 15 des phrases similaires à la question avec un seuil de 50%
      }
  )
  return retriever

retriever = retrieval(vector_database)
retriever_2 = retrieval(vector_database_2)
retriever_3 = retrieval(vector_database_3)
retriever_gnose = retrieval(vector_database_gnose)

def reranking(retriever):
  cohere_client = Client(api_key='iqcG53EsoG2Ps2nq1eHZzhJeoNQ4NwR8XliswG9i')
  compressor = CohereRerank(cohere_api_key='iqcG53EsoG2Ps2nq1eHZzhJeoNQ4NwR8XliswG9i', top_n = 3) #on prend les top 3 de ces top 15
  compression_retriever = ContextualCompressionRetriever(
     base_compressor=compressor, base_retriever=retriever
  )
  return compression_retriever

compression_retriever = reranking(retriever)
compression_retriever_2 = reranking(retriever_2)
compression_retriever_3 = reranking(retriever_3)
compression_retriever_gnose = reranking(retriever_gnose)


# fonction pour initialiser tokenizer
def initialize_tokenizer(model_name: str):
    """
    Initialize the tokenizer with the specified model_name.

    :param model_name: Name or path of the model for tokenizer initialization.
    :return: Initialized tokenizer.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.bos_token_id = 1  # Set beginning of sentence token id
    return tokenizer

# Initilialiser tokenizer & model
model_name = "HuggingFaceH4/zephyr-7b-beta"                               #"EleutherAI/gpt-j-6b"                                                #
tokenizer = initialize_tokenizer(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name)
model.generation_config.pad_token_id = model.generation_config.eos_token_id
# specifier stop token ids
stop_token_ids = [0]

def init_model(model, tokenizer):
    class CustomLLM(LLM):

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            model.to("cuda")
            # nouveau prompt de notre LLM
            prompt_template = """
            Use the following context elements to answer the question at the end.
            If the context does not contain any relevant information for the question,
            or there is no given context, simply say that you do not know the answer, do not attempt to invent an answer. Do not try to answer an other question.
            Answer like you were Jean-Luc Mélenchon, Gabriel Attal, Jordan Bardella who are candidates in the legislative elections 2024 in France.
            Always respond in the language in which the question is asked."""
            combined_prompt = f"{prompt_template}\n{prompt}"
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5)
            inputs = tokenizer(combined_prompt, return_tensors="pt").to("cuda")
            kwargs = dict(input_ids=inputs["input_ids"],
                          streamer=self.streamer,
                          max_new_tokens=512)

            # Générer le texte et récupérer la sortie générée
            generated_ids = model.generate(**kwargs)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if combined_prompt in generated_text:
                generated_text = generated_text.replace(combined_prompt, '').strip()
            generated_text = re.sub(r'Question:.*?Helpful Answer:.*?(?=\nQuestion:|\n*$)', '', generated_text, flags=re.DOTALL)
            return generated_text.strip()
        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()
    return llm
llm_for_response=init_model(model,tokenizer)     #LLM personnalisé pour générer la reponse des candidats


def init_model_gnose(model, tokenizer):
    class CustomLLM(LLM):

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            model.to("cuda")
            # nouveau prompt de notre LLM
            prompt_template = """
            Use the following context elements to answer the question at the end.
            If the context does not contain any relevant information for the question, or there is no given context,
            simply say that you do not know the answer, do not attempt to invent an answer. Do not try to answer an other question.
            Remember that you must respond in the language in which the question is asked."""
            combined_prompt = f"{prompt_template}\n{prompt}"
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5)
            inputs = tokenizer(combined_prompt, return_tensors="pt").to("cuda")
            kwargs = dict(input_ids=inputs["input_ids"],
                          streamer=self.streamer,
                          max_new_tokens=512)

            # Générer le texte et récupérer la sortie générée
            generated_ids = model.generate(**kwargs)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if combined_prompt in generated_text:
                generated_text = generated_text.replace(combined_prompt, '').strip()
            generated_text = re.sub(r'Question:.*?Helpful Answer:.*?(?=\nQuestion:|\n*$)', '', generated_text, flags=re.DOTALL)
            return generated_text.strip()
        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()
    return llm
llm_for_gnose=init_model_gnose(model,tokenizer)     #LLM personnalisé pour générer la reponse de GnoseIA


def init_model_for_overview(model, tokenizer):
    class CustomLLM(LLM):

        streamer: Optional[TextIteratorStreamer] = None

        def _call(self, prompt, stop=None, run_manager=None) -> str:
            model.to("cuda")
            self.streamer = TextIteratorStreamer(tokenizer, skip_prompt=True, timeout=5)
            inputs = tokenizer(prompt, return_tensors="pt").to("cuda")
            kwargs = dict(input_ids=inputs["input_ids"],
                          streamer=self.streamer,
                          max_new_tokens=512)

            # Générer le texte et récupérer la sortie générée
            generated_ids = model.generate(**kwargs)
            generated_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
            if prompt in generated_text:
                generated_text = generated_text.replace(prompt, '').strip()
            return generated_text.strip()

        @property
        def _llm_type(self) -> str:
            return "custom"

    llm = CustomLLM()
    return llm
llm_for_overview = init_model_for_overview(model, tokenizer)   #LLM personnalisé pour l'aperçu général du document


def combined_chunk(document:list)-> str:
    if len(document)>20:
        document = document[:21]
        doc = " ".join(chunk.page_content for chunk in document)
    else:
        doc = " ".join(chunk.page_content for chunk in document)
    return doc
# Créer un modèle LangChain pour générer un aperçu général
def create_overview():
    prompt_template = """
        En utilisant le document ci-dessous, générez un texte illustrant
        son titre s'il contient un titre, un très brève résumé de ce qu'il raconte et
        une proposition de trois questions auxquelles ce document peut répondre.
        Essayez de toujours générer des phrases complètes dans le texte généré. Répondez en langue avec laquelle le document est écrit.
        Par exemple: Ce document s'intitule (titre du document). Dans ce document, l'auteur parle de (brève résumé du document).
        Voici trois questions auxquelles ce document peut répondre (les trois questions proposées)".


        {text}


        Réponse:
        """
    prompt = PromptTemplate(template=prompt_template, input_variables=["text"])
    overview_chain = ( prompt | llm_for_overview )
    return overview_chain
def generate_overview(document:str, overview_chain)-> str:
    overview = overview_chain.invoke({"text": document})
    return overview
def overview_chain(document:list)-> str:
  combined_document = combined_chunk(document)
  overview_chain = create_overview()
  overview = generate_overview(combined_document, overview_chain)
  return overview


def get_variable_name(var):
    callers_local_vars = inspect.currentframe().f_back.f_locals.items()
    return [name for name, val in callers_local_vars if val is var]


# build conversational retrieval chain with memory (rag) using langchain
def Response_IA(query, chat_history, llm, retriever)-> tuple:
    text = get_variable_name(chat_history)[0]
    try:

        memory = ConversationBufferMemory(
            memory_key=text,
            return_messages=False,
        )
        qa_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=retriever,
            memory=memory,
            get_chat_history=lambda h: h,
        )
        result = qa_chain.invoke({'question' : query, text : chat_history})
        generated_text = result['answer']
        generated_text = generated_text.strip()
        chat_history.append((query, generated_text))

        return generated_text, chat_history

    except Exception as e:
        chat_history.append((query, e))
        return e, chat_history
    
    
def reference_for_retriever(results):          #fonction pour obtenir les références des programmes
  references = []
  reference = "Vous pouvez retrouver cette information ici: " + "\n"
  nombre_reference = 1
  for result in results:
    split = result.metadata["source"]
    splitted = split.split("/")
    source = splitted[-1]
    reference = reference + str(nombre_reference) + ") " + source + " , page "+ str(result.metadata["page"]+1) + "\n"
    nombre_reference += 1
  references.append(reference)
  return references


#initialisation de toutes les histoires de discussion, à exécuter une fois au début de discussion ou quand l'utilisateur réinitialise la discussion
chat_history_Front = []
chat_history_UDD = []
chat_history_RE = []
chat_history_gnose = []
chat_history_Gnose = []


def reponse_legislative(question):
    """
    Process user questions and files.

    :param questions: List of questions from the user.
    :param file_path: Path to the directory containing files.
    :return: List of responses from the AI and references for each question.
    """

    # Process each question
    # Get AI response for the question
    reponse_Front, chat_history_Front = Response_IA(question, chat_history_Front, llm_for_response, compression_retriever)
    print(reponse_Front)  #réponse de Front populaire

    reference_Front = compression_retriever.invoke(question)
    ref_Front = reference_for_retriever(reference_Front)
    print(ref_Front)    #réference dans le programme de Front populaire

    reponse_UDD, chat_history_UDD = Response_IA(question, chat_history_UDD, llm_for_response, compression_retriever_2)
    print(reponse_UDD)  #réponse de UDD

    reference_UDD = compression_retriever_2.invoke(question)
    ref_UDD = reference_for_retriever(reference_UDD)
    print(ref_UDD)       #réference dans le programme UDD

        
    reponse_RE, chat_history_RE = Response_IA(question, chat_history_RE, llm_for_response, compression_retriever_3)
    print(reponse_RE)    #réponse de Renaissance

    reference_RE = compression_retriever_3.invoke(question)
    ref_RE = reference_for_retriever(reference_RE)
    print(ref_RE)        #réference dans le programme de Renaissance

    answer_gnose, chat_history_Gnose = Response_IA(question, chat_history_Gnose, llm_for_gnose, compression_retriever_gnose)
    print(answer_gnose)    #réponse de gnoseIA

    Reference_gnose = compression_retriever_gnose.invoke(question)
    Ref_gnose = reference_for_retriever(Reference_gnose)
    print(Ref_gnose)       #réference dans le corpus gnoseia

    # Store the response and references for the question
    responses_and_references = {
       "Front": {"question": question, "reponse": reponse_Front, "references": ref_Front},
       "UUD": {"question": question, "reponse": reponse_UDD, "references": ref_UDD},
       "Renaissance": {"question": question, "reponse": reponse_RE, "references": ref_RE},
       "Gnoseia": {"question": question, "reponse": answer_gnose, "references": Ref_gnose},
    }

    return responses_and_references

def reponse_corpus_gnoseia(question):    
    reponse_gnose, chat_history_gnose = Response_IA(question, chat_history_gnose, llm_for_gnose, compression_retriever_gnose)
    print(reponse_gnose) #réponse de gnoseIA 

    reference_gnose = compression_retriever_gnose.invoke(question)
    ref_gnose = reference_for_retriever(reference_gnose)
    print(ref_gnose)   #réference dans le corpus de gnoseia

    # Store the response and references for the question
    responses_and_references = {"question": question, "reponse": reponse_gnose, "references": ref_gnose}

    return responses_and_references

def reponse_gnoseia(question, file_path):
    # Load documents from the specified path
    documents = load_document(file_path)
    
    # Split documents into chunks
    chunks = split_document(documents)

    vector_database_entree, collection_entree = init_chroma(chunks, 4)
    retriever_entree = retrieval(vector_database_entree)

    compression_retriever_entree = reranking(retriever_entree)

    reponse_gnose, chat_history_gnose = Response_IA(question, chat_history_gnose, llm_for_gnose, compression_retriever_entree)
    print(reponse_gnose) #réponse de gnoseIA sur les documents ajoutés par l'utilisateur

    reference_gnose = compression_retriever_entree.invoke(question)
    ref_gnose = reference_for_retriever(reference_gnose)
    print(ref_gnose)   #réference dans les documents ajoutés

    overview = overview_chain(chunks)
    print(overview)    #aperçu des documents ajoutés
    
    responses_overview_and_references = {"question": question, "reponse": reponse_gnose, "overview": overview, "references": ref_gnose}

    return responses_overview_and_references