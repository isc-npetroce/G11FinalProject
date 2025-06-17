# hf_gradio_ai_app.py
#
# To run this script:
# 1. Create a .env file in the same directory with your OPENAI_API_KEY.
#    Example .env file content:
#    OPENAI_API_KEY="sk-yourActualOpenAIapiKeyGoesHere"
# 2. Install the required packages:
#    pip install gradio langchain openai langchain_openai python-dotenv
# 3. Run the script from your terminal:
#    python hf_gradio_ai_app.py
#
# The script will output a local URL and potentially a public Gradio link.

import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from langchain.schema.output_parser import StrOutputParser
import os
from dotenv import load_dotenv
from data_loader import load_vector_db

# --- Load environment variables from .env file ---
load_dotenv()

# --- Global variables and Initial Setup ---
OPENAI_API_KEY_GLOBAL = os.getenv("OPENAI_API_KEY")
LANGCHAIN_LLM = None
LANGCHAIN_PROMPT_TEMPLATE = None
INITIAL_AI_SETUP_MESSAGE = "" # To store status/error from initial setup

# https://colab.research.google.com/drive/1G5YiHIDZzRG9AcUMiNd9MITowaNHUKyK?usp=sharing
def initialize_ai_components():
    """
    Initializes LangChain components (LLM and prompt template) using the API key
    from environment variables. Updates global variables and sets a status message.
    """
    global LANGCHAIN_LLM, LANGCHAIN_PROMPT_TEMPLATE, OPENAI_API_KEY_GLOBAL, INITIAL_AI_SETUP_MESSAGE
    
    if not OPENAI_API_KEY_GLOBAL:
        INITIAL_AI_SETUP_MESSAGE = "<p style='color:red; font-weight:bold;'>ERROR: OpenAI API Key not found. Please ensure it's in your .env file or environment variables.</p>"
        print("ERROR: OpenAI API Key not found. Make sure it's in your .env file or environment.")
        return False # Indicate failure
    
    try:
        # Initialize the LangChain LLM (OpenAI model)
        LANGCHAIN_LLM = ChatOpenAI(openai_api_key=OPENAI_API_KEY_GLOBAL, model_name="gpt-4o-mini")
        
        # Define the prompt template for the LLM
        prompt_template_str = """
        You are a helpful, friendly, and insightful AI assistant.
        Your task is to summarize the clinical notes for one or many patients (if the name is not specified, it if for many). The user query might also specify additional instructions -- if there is a different task requested, answer that instead of simply summarizing the Clinical Notes. 
        If there are no clinical notes provided, let the user know that there is no info for the provided input and ask them to check if valid details were provided. 

        User query: {user_query}

        {field_name}: {field_value}

        Clinical Notes: {clinical_notes}

        AI Summary:
        """
        LANGCHAIN_PROMPT_TEMPLATE = ChatPromptTemplate.from_template(prompt_template_str)
        
        INITIAL_AI_SETUP_MESSAGE = "<p style='color:green; font-weight:bold;'>AI Components Initialized Successfully! Ready to chat.</p>"
        print("AI Components Initialized Successfully!")
        return True # Indicate success
    except Exception as e:
        INITIAL_AI_SETUP_MESSAGE = f"<p style='color:red; font-weight:bold;'>ERROR: Failed to initialize AI components. Error: {str(e)}. Please check your API key and model access.</p>"
        LANGCHAIN_LLM = None
        LANGCHAIN_PROMPT_TEMPLATE = None
        print(f"ERROR: Failed to initialize AI components: {str(e)}")
        return False # Indicate failure

# --- Attempt to initialize AI components when the script loads ---
AI_INITIALIZED_SUCCESSFULLY = initialize_ai_components()


def get_rag_response(input_map):
    user_query = input_map['user_query']
    field_name = input_map['field_name']
    field_value = input_map['field_value']

    collection = load_vector_db()

    joined_docs = ""

    if user_query and field_value:
        retrieval_results = collection.query(
            query_texts=user_query,
            n_results=10,
            where={field_name: field_value},
        )
        joined_docs = '\n\n'.join(retrieval_results['documents'][0])

    elif not field_value:
        retrieval_results = collection.query(
            query_texts=user_query,
            n_results=10,
        )
        joined_docs = '\n\n'.join(retrieval_results['documents'][0])

    else:
        retreival_results1 = collection.get(
            where={field_name: field_value},
        )
        joined_docs = '\n\n'.join(str(doc) for doc in retreival_results1['documents'])

    print("User query: ", user_query)

    print("\nJoined docs: ", joined_docs)

    return joined_docs


def get_user_query(input_map):
    return input_map['user_query']


def get_field_name(input_map):
    return input_map['field_name']


def get_field_value(input_map):
    return input_map['field_value']


def ai_chat_response_function(user_query, _, field_name, field_value):
    """
    This is the core function called by Gradio's ChatInterface.
    """
    if not AI_INITIALIZED_SUCCESSFULLY or not LANGCHAIN_LLM or not LANGCHAIN_PROMPT_TEMPLATE:
        # Use the globally set error message from initialization
        # Clean up HTML for plain error string if needed, or pass raw if Markdown supports it
        error_msg_text = INITIAL_AI_SETUP_MESSAGE.replace("<p style='color:red; font-weight:bold;'>", "").replace("</p>", "")
        return f"ERROR: AI is not ready. Status: {error_msg_text}"

    # Proceed with generating response if components are ready
    try:
        # Create the LangChain chain
        chain = ( 
            {"clinical_notes": get_rag_response, "user_query": get_user_query, "field_name": get_field_name, "field_value": get_field_value} 
            | LANGCHAIN_PROMPT_TEMPLATE 
            | LANGCHAIN_LLM 
            | StrOutputParser()
        )
        
        # Invoke the chain with the user's input
        input_map = {
            'user_query': user_query,
            'field_name': field_name,
            'field_value': field_value 
        }
        ai_response = chain.invoke(input_map)
        
        # Return the content of the AI's response
        return ai_response
    except Exception as e:
        print(f"Error during LangChain invocation: {e}") # Log for server-side debugging
        return f"Sorry, an error occurred while trying to get a response: {str(e)}"


with gr.Blocks(theme=gr.themes.Soft(primary_hue=gr.themes.colors.blue, secondary_hue=gr.themes.colors.sky), title="AI Chatbot (Gradio)") as gradio_app:
    gr.Markdown(
        """
        # 🤖 AI Clinical Notes Summarizer. 
        Powered by OpenAI's `gpt-4o-mini` model.
        """
    )

    # Display the initial AI setup status
    gr.Markdown(INITIAL_AI_SETUP_MESSAGE)
    
    gr.Markdown("---") # Visual separator
    gr.Markdown("## Chat Interface")

    # Gradio ChatInterface for the main chat functionality
    chat_interface_component = gr.ChatInterface(
        fn=ai_chat_response_function, # The function that handles chat logic
        chatbot=gr.Chatbot(
            height=350,
            show_label=False,
            placeholder="AI's responses will appear here." if AI_INITIALIZED_SUCCESSFULLY else "AI is not available. Check setup status above.",
            avatar_images=("https://raw.githubusercontent.com/svgmoji/svgmoji/main/packages/svgmoji__openmoji/svg/1F468-1F3FB-200D-1F9B0.svg", "https://raw.githubusercontent.com/gradio-app/gradio/main/gradio/icons/huggingface-logo.svg"),
            type='messages'
        ),
        textbox=gr.Textbox(
            placeholder="Enter your query",
            show_label=False,
            # Disable textbox if AI did not initialize successfully
            interactive=AI_INITIALIZED_SUCCESSFULLY
        ),
        submit_btn="➡️ Send",
        title=None,
        autofocus=True,
        additional_inputs=[
            gr.Dropdown(
                choices=['Patient ID', 'Patient Name', 'Encounter ID'],
                label="Select Field Name",
                value="Patient Name",  # Default selected value
                interactive=AI_INITIALIZED_SUCCESSFULLY
            ),
            gr.Textbox(
                placeholder="Enter Field Value",
                show_label=False,
                # Disable textbox if AI did not initialize successfully
                interactive=AI_INITIALIZED_SUCCESSFULLY
            )
        ],
    )
    
    if not AI_INITIALIZED_SUCCESSFULLY:
        pass


# --- Main execution block to launch the Gradio app ---
if __name__ == '__main__':
    print("Attempting to launch Gradio App...")
    if not OPENAI_API_KEY_GLOBAL:
        print("WARNING: OpenAI API Key was not found in environment variables or .env file.")
        print("The application UI will launch, but AI functionality will be disabled.")
        print("Please create a .env file with your OPENAI_API_KEY.")
    
    gradio_app.launch(share=True, debug=True)