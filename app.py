# Import os to set API key (can sub this out for other LLM providers)
import os
# Import the API key
from dotenv import load_dotenv
# IMport OpenAI as main LLM service
from langchain_openai import OpenAI
# Bring in streamlit for UI/app interface
import streamlit as st

# Import PDF document loaders...there's other ones as well!
from langchain_community.document_loaders import PyPDFLoader
# Import the sentence transformer embedding function
from langchain_community.embeddings.sentence_transformer import (SentenceTransformerEmbeddings,)
# Import chroma as the vector store
from langchain_community.vectorstores import Chroma

# Import vector store stuff
from langchain.agents.agent_toolkits import (
    create_vectorstore_agent,
    VectorStoreToolkit,
    VectorStoreInfo,
)

# Set up the API Key
load_dotenv()

# Create instance of OpenAI LLM (the model)

llm = OpenAI(temperature=0.9)

# Create and load PDF Loader
loader = PyPDFLoader("cnn research paper.pdf")

# Split pages from pdf
pages = loader.load_and_split()

# First, create the open-source embedding function
embedding_function = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

# Load documents into vector database aka ChromaDB
store = Chroma.from_documents(pages, embedding_function, collection_name="research_papers")

# Create vectorstore info obhject - metadata repo?
vectorstore_info = VectorStoreInfo(
    name="research_papers",
    description="a research paper as a pdf",
    vectorstore=store,
)

# Convert the document store into a langchain toolkit
toolkit =  VectorStoreToolkit(vectorstore_info=vectorstore_info)

# Add the toolkit to an end-to-end LC
agent_executor = create_vectorstore_agent(
    llm=llm,
    toolkit=toolkit,
    verbose=True,
)


# Create a text input box for the user
prompt = st.text_input("Input your prompt here")

# If the user hits enter
if prompt:
    # Then pass the prompt to the LLM
    # response = llm(prompt)
    
    # Swap out the raw llm for a document agent
    response = agent_executor.run(prompt)
    
    # ...and write it out to the screen
    st.write(response)
    
    # With a streamlit expander
    with st.expander("Document Similarity Search"):
        # Find the relevant pages
        search = store.similarity_search_with_score(prompt)
        
        # Write out the first
        st.write(search[0][0].page_content)