import os
import json
import traceback
import pandas as pd
from dotenv import load_dotenv
from src.mcqgenerator.utils import read_file,get_table_data
from src.mcqgenerator.logger import logging

#imporing necessary packages packages from langchain
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from langchain_classic.chains import LLMChain
from langchain_classic.chains import SequentialChain


# Load environment variables from the .env file
load_dotenv()

# Access the environment variables - support both OpenAI and Groq
openai_key = os.getenv("OPENAI_API_KEY")
groq_key = os.getenv("GROQ_API_KEY")

# List of Groq models to try (in order of preference)
# Check current models at: https://console.groq.com/docs/models
GROQ_MODELS = [
    "llama-3.3-70b-versatile",      # Latest Llama 3.3 model
    "llama-3.1-8b-instant",         # Fast 8B model
    "llama-3.1-70b-versatile",      # 70B model
    "gemma2-9b-it",                 # Gemma model
    "llama3-8b-8192",               # Alternative 8B model
]

# Use Groq if available (free tier), otherwise fall back to OpenAI
if groq_key:
    llm = None
    last_error = None
    
    # Try each model until one works
    for model_name in GROQ_MODELS:
        try:
            llm = ChatGroq(groq_api_key=groq_key, model_name=model_name, temperature=0.7)
            logging.info(f"Using Groq API (free tier) with model: {model_name}")
            break
        except Exception as e:
            last_error = e
            logging.warning(f"Model {model_name} failed, trying next...")
            continue
    
    if llm is None:
        error_msg = f"All Groq models failed. Last error: {str(last_error)}\n"
        error_msg += "Please check available models at: https://console.groq.com/docs/models\n"
        error_msg += "Or update GROQ_MODELS list in MCQGenerator.py with current model names."
        raise ValueError(error_msg)
        
elif openai_key:
    llm = ChatOpenAI(openai_api_key=openai_key, model_name="gpt-3.5-turbo", temperature=0.7)
    logging.info("Using OpenAI API")
else:
    raise ValueError("Please set either GROQ_API_KEY or OPENAI_API_KEY in your .env file")

template="""
Text:{text}
You are an expert MCQ maker. Given the above text, it is your job to \
create a quiz  of {number} multiple choice questions for {subject} students in {tone} tone. 
Make sure the questions are not repeated and check all the questions to be conforming the text as well.
Make sure to format your response like  RESPONSE_JSON below  and use it as a guide. \
Ensure to make {number} MCQs
### RESPONSE_JSON
{response_json}

"""

quiz_generation_prompt = PromptTemplate(
    input_variables=["text", "number", "subject", "tone", "response_json"],
    template=template)


quiz_chain=LLMChain(llm=llm,prompt=quiz_generation_prompt,output_key="quiz",verbose=True)


template2="""
You are an expert english grammarian and writer. Given a Multiple Choice Quiz for {subject} students.\
You need to evaluate the complexity of the question and give a complete analysis of the quiz. Only use at max 50 words for complexity analysis. 
if the quiz is not at per with the cognitive and analytical abilities of the students,\
update the quiz questions which needs to be changed and change the tone such that it perfectly fits the student abilities
Quiz_MCQs:
{quiz}

Check from an expert English Writer of the above quiz:
"""


quiz_evaluation_prompt=PromptTemplate(input_variables=["subject", "quiz"], template=template2)

review_chain=LLMChain(llm=llm, prompt=quiz_evaluation_prompt, output_key="review", verbose=True)


# This is an Overall Chain where we run the two chains in Sequence
generate_evaluate_chain=SequentialChain(chains=[quiz_chain, review_chain], input_variables=["text", "number", "subject", "tone", "response_json"],
                                        output_variables=["quiz", "review"], verbose=True,)
