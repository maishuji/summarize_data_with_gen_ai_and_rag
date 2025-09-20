import os

# Disable Chroma telemetry before importing Chroma
os.environ["CHROMA_TELEMETRY_IMPLEMENTATION"] = "none"
os.environ["CHROMA_TELEMETRY"] = "none"

import torch
import logging
import re
from time import sleep
from langchain_core.prompts import (
    PromptTemplate,
)  # Updated import per deprecation notice
from langchain.chains import RetrievalQA
from huggingface_hub.errors import HfHubHTTPError, BadRequestError

from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_ibm import WatsonxLLM
from langchain_core.prompts import (
    ChatPromptTemplate,
    MessagesPlaceholder,
)
from langchain_core.messages import BaseMessage 

from langchain_community.vectorstores import Chroma  # New import path
from langchain_community.embeddings import (
    HuggingFaceInstructEmbeddings,
)  # New import path
from langchain_community.document_loaders import PyPDFLoader  # New import path
from huggingface_hub import InferenceClient
from langchain.chains.combine_documents import (
    create_stuff_documents_chain,
)  # Import the missing function
from langchain.chains import create_retrieval_chain  # Import retrieval chain creator
from langchain_core.runnables import RunnableLambda
from dotenv import load_dotenv

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Check for GPU availability and set the appropriate device for computation.
DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"

# Global variables
conversation_retrieval_chain = None
chat_history = []
llm_hub = None
embeddings = None

# Load environment variables from .env file
load_dotenv()
WATSONX_URL = os.getenv("WATSONX_URL")
PROJECT_ID = os.getenv("PROJECT_ID")
WATSONX_API_KEY = os.getenv("WATSON_API_KEY")
HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")


model_id = "cnicu/t5-small-booksum"

client = InferenceClient(model=model_id, token=HUGGINGFACEHUB_API_TOKEN, timeout=60)

_CAPS = "A-ZÉÈÀÂÎÔÙÛÇÄËÏÖÜŸ"            # capital letters incl. French accents
_LOWER = "a-zà-öù-ÿ"                    # lowercase incl. accents

def clean_pdf_text(t: str) -> str:
    if not isinstance(t, str):
        return t

    # 1) Turn stray 'n' that marks a new paragraph/line into a real newline:
    #    start-of-line or after whitespace/punct, then 'n' then Capital → newline
    t = re.sub(rf"(^|[ \t\r\f\v,;:—–\-•\(\)\[\]{{}}])n(?=[{_CAPS}])", r"\1\n", t)

    # 2) Fix words glued across old line breaks: 'pournÉtudier' → 'pour Étudier'
    t = re.sub(rf"([{_LOWER}])n([{_CAPS}])", r"\1 \2", t)

    # 3) Collapse excessive blank lines
    t = re.sub(r"\n{3,}", "\n\n", t)

    return t

def clean_answer(ans: str) -> str:
    if not isinstance(ans, str):
        return ans
    # same rules but a tad more conservative on spaces
    ans = re.sub(rf"(^|[ \t,;:—–\-•\(\)\[\]{{}}])n(?=[{_CAPS}])", r"\1\n", ans)
    ans = re.sub(rf"([{_LOWER}])n([{_CAPS}])", r"\1 \2", ans)
    ans = re.sub(r"\n{3,}", "\n\n", ans)
    return ans

def _extract_summary(res) -> str:
    # HF Inference may return:
    # - str
    # - pydantic/dataclass-like object with `.summary_text` or `.generated_text`
    # - dict/list variants
    if isinstance(res, str):
        return res
    if hasattr(res, "summary_text"):
        return res.summary_text
    if hasattr(res, "generated_text"):
        return res.generated_text
    if isinstance(res, dict):
        return str(res.get("summary_text") or res.get("generated_text") or res)
    if isinstance(res, list) and res:
        item = res[0]
        if isinstance(item, dict):
            return str(item.get("summary_text") or item.get("generated_text") or item)
        if hasattr(item, "summary_text"):
            return item.summary_text
        if hasattr(item, "generated_text"):
            return item.generated_text
        return str(item)
    return str(res)

def _to_text(x) -> str:
    # 1) ChatPromptValue-like
    if hasattr(x, "to_string") and callable(getattr(x, "to_string")):
        return x.to_string()
    if hasattr(x, "to_messages") and callable(getattr(x, "to_messages")):
        try:
            msgs = x.to_messages()
            return "\n".join(getattr(m, "content", "") for m in msgs if hasattr(m, "content"))
        except Exception:
            pass
    # NEW: many LC prompt values expose a `.messages` list directly
    if hasattr(x, "messages"):
        try:
            return "\n".join(getattr(m, "content", "") for m in x.messages if hasattr(m, "content"))
        except Exception:
            pass

    # 2) Single LC message
    if isinstance(x, BaseMessage):
        return x.content

    # 3) List/tuple of messages
    if isinstance(x, (list, tuple)) and x and isinstance(x[0], BaseMessage):
        return "\n".join(getattr(m, "content", str(m)) for m in x)

    # 4) Dict from upstream runnables
    if isinstance(x, dict):
        for k in ("input", "question", "prompt", "text", "query", "context"):
            if k in x and x[k] is not None:
                return str(x[k])

    # 5) Fallback
    return str(x)



def _summarize(text: str) -> str:
    # Hard guard against huge inputs (T5-small chokes on long text)
    MAX_CHARS = 4000   # tune as needed
    text = _to_text(text)
    if len(text) > MAX_CHARS:
        text = text[:MAX_CHARS]

    last_err = None
    for attempt in range(3):
        try:
            res = client.summarization(
                model="cnicu/t5-small-booksum",
                text=text,
            )
            # Extract string (as we already did before)
            return _extract_summary(res)
        except BadRequestError as e:
            # 400-level: usually parameter/size issues; break immediately
            raise
        except HfHubHTTPError as e:
            last_err = e
            # 5xx: transient — backoff and retry
            sleep(1.5 * (2 ** attempt))
    # After retries, bubble up the last error
    raise last_err

# Function to initialize the language model and its embeddings
def init_llm():
    global llm_hub, embeddings

    logger.info("Initializing WatsonxLLM and embeddings...")

    # Llama Model Configuration
    MODEL_ID = "meta-llama/llama-3-3-70b-instruct"
    PROJECT_ID = os.getenv("PROJECT_ID")

    # Use the same parameters as before:
    #   MAX_NEW_TOKENS: 256, TEMPERATURE: 0.1
    model_parameters = {
        # "decoding_method": "greedy",
        "max_new_tokens": 256,
        "temperature": 0.1,
    }

    # Initialize Llama LLM using the updated WatsonxLLM API
    # llm_hub = WatsonxLLM(
    #    apikey=Watsonx_API,
    #    model_id=MODEL_ID,
    #    url=WATSONX_URL,
    #    project_id=PROJECT_ID,
    #    params=model_parameters
    # )

    # Using HuggingFaceHub LLM
    llm_hub = RunnableLambda(_summarize)

    logger.info("llm_hub type: %s", type(llm_hub))
    logger.info("Using HF task: summarization")

    logger.debug("WatsonxLLM initialized: %s", llm_hub)

    # Initialize embeddings using a pre-trained model to represent the text data.
    embeddings = HuggingFaceInstructEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={"device": DEVICE},
    )

    logger.debug("Embeddings initialized with model device: %s", DEVICE)


def process_document(document_path):
    global conversation_retrieval_chain
    global llm_hub  # Ensure llm_hub is accessible

    logger.info("Loading document from path: %s", document_path)
    # Load the document
    loader = PyPDFLoader(document_path)
    documents = loader.load()
    for d in documents:
        d.page_content = clean_pdf_text(d.page_content)
    logger.debug("Loaded %d document(s)", len(documents))

    # Split the document into chunks, set chunk_size=1024, and chunk_overlap=64. assign it to variable text_splitter
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1024, chunk_overlap=64)
    texts = text_splitter.split_documents(documents)
    logger.debug("Document split into %d text chunks", len(texts))

    # Create an embeddings database using Chroma from the split text chunks.
    logger.info("Initializing Chroma vector store from documents...")

    # Build the QA chain, which utilizes the LLM and retriever for answering questions.
    db = Chroma.from_documents(texts, embedding=embeddings)
    retriever = db.as_retriever(
        search_type="mmr", search_kwargs={"k": 6, "lambda_mult": 0.25}
    )

    # 3) Build the “stuff” doc chain + retrieval chain
    logger.info("Creating retrieval chain...")
    prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "You are a helpful assistant. Use the provided context to answer the user's question clearly and concisely.",
            ),
            MessagesPlaceholder("chat_history"),
            ("human", "Question: {input}\n\nContext:\n{context}\n\nAnswer:"),
        ]
    )
    logger.info("Prompt template created: %s", prompt)

    doc_chain = create_stuff_documents_chain(llm_hub, prompt)  # Use llm_hub as the LLM
    conversation_retrieval_chain = create_retrieval_chain(retriever, doc_chain)
    if conversation_retrieval_chain is None:
        logger.error("Failed to create conversation_retrieval_chain.")
        raise RuntimeError("Failed to create conversation_retrieval_chain.")

    logger.info("Retrieval chain created successfully.")


# Function to process a user prompt
def process_prompt(prompt):
    global conversation_retrieval_chain
    global chat_history

    logger.info("Processing prompt: %s", prompt)

    if conversation_retrieval_chain is None:
        logger.error(
            "conversation_retrieval_chain is not initialized. Please call process_document() first."
        )
        raise RuntimeError(
            "conversation_retrieval_chain is not initialized. Please call process_document() with a valid document path before processing prompts."
        )

    # Query the model using the new .invoke() method
    output = conversation_retrieval_chain.invoke(
        {"input": prompt, "chat_history": chat_history}
    )
    logger.info("Model invoked successfully.")
    answer = output.get("answer") or ""

    logger.info("Answer: %s", answer)
    logger.debug("Model response: %s", answer)

    # Update the chat history
    chat_history.append((prompt, answer))
    logger.debug("Chat history updated. Total exchanges: %d", len(chat_history))

    # Return the model's response
    # answer = clean_answer(answer)
    return answer


# Initialize the language model
init_llm()
logger.info("LLM and embeddings initialization complete.")
