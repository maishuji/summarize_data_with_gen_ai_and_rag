# RAG --- Retrieval-Augmented Generation

**Retrieval-Augmented Generation (RAG)** is a technique that enhances
Large Language Models (LLMs) by allowing them to retrieve external,
up-to-date information before generating responses.\
This approach improves accuracy, reduces hallucinations, and provides
responses backed by real sources.

------------------------------------------------------------------------

## Challenges of LLMs

Although LLMs are powerful, they have some limitations:

-   **Lack of sources** --- They generate text based on patterns from
    training data, without citing where information comes from.\
-   **Outdated knowledge** --- Their knowledge is frozen at the time
    they were trained and does not automatically update with new
    information.

------------------------------------------------------------------------

## How RAG Works

Instead of relying solely on the knowledge built into the LLM, **RAG
adds a retrieval layer** that pulls in relevant external content at
runtime.

This content can come from:

-   **Open sources** --- such as the internet, public APIs, or knowledge
    bases\
-   **Closed sources** --- such as private document collections, company
    data, or internal knowledge repositories

The retrieved data is provided as context to the LLM, which then
generates more **accurate, relevant, and verifiable responses.**

------------------------------------------------------------------------

## Benefits of RAG

-   **Current information** --- Incorporates the latest data into
    responses\
-   **Context-aware answers** --- Tailors outputs to specific documents
    or datasets\
-   **Explainable outputs** --- Can cite the retrieved sources\
-   **Reduced hallucinations** --- Lessens the chance of the LLM making
    up information
