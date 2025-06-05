import streamlit as st
import os
from dotenv import load_dotenv

from langchain_groq import ChatGroq
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate


load_dotenv()


groq_api_key_check = os.getenv("GROQ_API_KEY")

# LangSmith tracking 
os.environ["LANGCHAIN_API_KEY"] = os.getenv("LANGCHAIN_API_KEY")
os.environ["LANGCHAIN_TRACING_V2"] = "true"
os.environ["LANGCHAIN_PROJECT"] = "Q&A Chatbot with Groq"


prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant. Please respond to the user queries."),
    ("user", "Question: {question}")
])


def generate_response(question, model_name, temperature, max_tokens):
    llm = ChatGroq(
            model=model_name,
            groq_api_key=os.getenv("GROQ_API_KEY"),
            temperature=temperature,
            max_tokens=max_tokens
        )
        
    output_parser = StrOutputParser()
    chain = prompt | llm | output_parser
    response = chain.invoke({"question": question})
    return response  
        
    

# Streamlit UI
st.title("Q&A Chatbot for OS models from Ollama with Groq")
st.sidebar.header("Settings")

model_name = st.sidebar.selectbox("Select an OS Model", ["Gemma2-9b-It", "llama3-8b-8192"])

temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=1.0, value=0.7, help="Controls the randomness of the output. Lower values are more deterministic, higher values are more creative.")
max_tokens = st.sidebar.slider("Max Tokens", min_value=50, max_value=300, value=150, help="The maximum number of tokens (words/pieces of words) in the generated response.")


st.write("Go ahead and ask any questions.")
user_input = st.text_input("Your Question:", placeholder="Type your question here...")

if user_input:    
    with st.spinner("Thinking..."):
        response = generate_response(user_input, model_name, temperature, max_tokens)
    st.write(response)

else:
    st.write("Please provide a query to get started!")

