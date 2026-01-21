import streamlit as st
import os
import tempfile

# --- IMPORTATIONS LANGCHAIN ---
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import Ollama
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="Mon Assistant de Cours", page_icon="🤖")
st.title("🤖 Spaceflight I(A)nstitute")

# Configuration Proxy (si nécessaire)
os.environ["NO_PROXY"] = "localhost,127.0.0.1"

# --- FONCTION DE CHARGEMENT (MISE EN CACHE) ---
# @st.cache_resource est CRUCIAL : il empêche de tout recharger à chaque question
@st.cache_resource
def initialize_rag_chain(pdf_file_path):
    # 1. Chargement
    loader = PyPDFLoader(pdf_file_path)
    pages = loader.load()
    
    # 2. Découpage
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    chunks = text_splitter.split_documents(pages)
    
    # 3. Vectorisation
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(documents=chunks, embedding=embeddings)
    retriever = vectorstore.as_retriever()
    
    # 4. Modèle et Chaîne
    llm = Ollama(model="mistral")
    
    system_prompt = (
        "Tu es un professeur assistant pédagogique. "
        "Utilise le contexte suivant pour répondre à la question de l'étudiant."
        "Si tu ne trouves pas la réponse dans le contexte, dis-le sèchement."
        "\n\n"
        "{context}"
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            ("human", "{input}"),
        ]
    )
    
    question_answer_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, question_answer_chain)
    
    return rag_chain

# --- INTERFACE UTILISATEUR (SIDEBAR) ---
with st.sidebar:
    st.header("Configuration")
    # Option pour uploader le fichier directement dans l'interface
    uploaded_file = st.file_uploader("Chargez votre cours (PDF)", type="pdf")

# --- INITIALISATION DU RAG ---
if uploaded_file is not None:
    # On doit sauvegarder le fichier uploadé temporairement pour que PyPDFLoader puisse le lire
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.getvalue())
        tmp_file_path = tmp_file.name

    with st.spinner("Analyse du document en cours..."):
        try:
            rag_chain = initialize_rag_chain(tmp_file_path)
            st.success("Assistant prêt !")
        except Exception as e:
            st.error(f"Erreur : {e}")
else:
    # Optionnel : Utiliser un fichier par défaut si aucun upload
    default_file = "mon_cours.pdf"
    if os.path.exists(default_file):
        rag_chain = initialize_rag_chain(default_file)
    else:
        st.info("Veuillez charger un fichier PDF dans la barre latérale.")
        st.stop() # Arrête le script ici si pas de fichier

# --- GESTION DE L'HISTORIQUE ---
if "messages" not in st.session_state:
    st.session_state.messages = []

# Afficher les anciens messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# --- ZONE DE CHAT ---
if prompt := st.chat_input("Posez votre question sur le cours..."):
    # 1. Afficher le message de l'utilisateur
    with st.chat_message("user"):
        st.markdown(prompt)
    # Ajouter à l'historique
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 2. Générer la réponse
    with st.chat_message("assistant"):
        with st.spinner("Réflexion..."):
            response = rag_chain.invoke({"input": prompt})
            answer = response["answer"]
            st.markdown(answer)
    
    # Ajouter la réponse à l'historique
    st.session_state.messages.append({"role": "assistant", "content": answer})
    