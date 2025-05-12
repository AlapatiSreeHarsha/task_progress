import streamlit as st
import os
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize session state for storing uploaded files and chat history
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True
    )

# Set page config
st.set_page_config(
    page_title="File Analysis Agent",
    page_icon="🤖",
    layout="wide"
)

# Initialize Gemini model
def initialize_model():
    try:
        model = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            google_api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7,
            convert_system_message_to_human=True
        )
        return model
    except Exception as e:
        st.error(f"Error initializing model: {str(e)}")
        return None

# File handling tools
@tool
def read_file(file_path: str) -> str:
    """Read the contents of a file."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file."""
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List all files in a directory."""
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

def create_agent(model):
    tools = [read_file, write_file, list_files]
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that helps analyze and interact with files and folders.
        You can read files, write to files, and list directory contents.
        
        When files are uploaded:
        1. First, list all available files
        2. Then, read and analyze the contents of key files (like README.md, main files, etc.)
        3. Provide a comprehensive overview of the project
        
        When asked to summarize:
        1. Analyze the project structure
        2. Identify main components and their purposes
        3. Explain key functionalities
        4. Highlight important features
        
        Always be proactive in analyzing files and providing insights.
        When writing to files, ensure to maintain code quality and follow best practices.
        Provide clear explanations of your actions and findings.
        
        Remember to reference past conversations when relevant to provide context-aware responses."""),
        ("human", "{input}")
    ])

    return model, tools, prompt

def analyze_uploaded_files(saved_files):
    """Initial analysis of uploaded files"""
    analysis = []
    for file_name, file_path in saved_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
                analysis.append(f"File: {file_name}\nContent Preview: {content[:200]}...\n")
        except Exception as e:
            analysis.append(f"File: {file_name}\nError reading file: {str(e)}\n")
    return "\n".join(analysis)

def save_uploaded_files(uploaded_files):
    temp_dir = tempfile.mkdtemp()
    saved_files = {}
    
    for uploaded_file in uploaded_files:
        file_path = os.path.join(temp_dir, uploaded_file.name)
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        saved_files[uploaded_file.name] = file_path
    
    return temp_dir, saved_files

def process_with_agent(model, tools, prompt, input_text):
    """Process the input with the agent and tools"""
    try:
        # First, let's list the files
        file_list = list_files(os.path.dirname(next(iter(st.session_state.uploaded_files.values()))))
        
        # Then, read the contents of each file
        file_contents = {}
        for file_name, file_path in st.session_state.uploaded_files.items():
            content = read_file(file_path)
            file_contents[file_name] = content
        
        # Check if this is an edit request
        if "change" in input_text.lower() or "edit" in input_text.lower() or "replace" in input_text.lower():
            for file_name, file_path in st.session_state.uploaded_files.items():
                if file_name.lower() in input_text.lower():
                    current_content = read_file(file_path)
                    
                    # Create a specific prompt for editing
                    edit_prompt = f"""You are a file editing assistant. Your task is to modify the content of {file_name}.
                    
Current content:
{current_content}

Requested change: {input_text}

Please provide ONLY the complete updated content with the requested changes. Do not include any explanations or additional text."""

                    # Get the updated content
                    updated_content = model.invoke(edit_prompt).content
                    
                    # Write the changes back to the file
                    write_file(file_path, updated_content)
                    
                    # Update memory with the edit
                    st.session_state.memory.save_context(
                        {"input": input_text},
                        {"output": f"Made changes to {file_name}"}
                    )
                    
                    # Return the confirmation with before/after
                    return f"""I've made the requested changes to {file_name}.

Before:
{current_content}

After:
{updated_content}

Changes have been saved successfully."""
        
        # For non-edit requests, use the regular prompt with memory
        context = f"""Available files:
{file_list}

File contents:
{file_contents}

User question: {input_text}"""
        
        # Get chat history from memory
        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        
        # Format chat history for the prompt
        formatted_history = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in chat_history
        ])
        
        # Get response from model with chat history
        response = model.invoke(prompt.format(
            input=f"{formatted_history}\n\n{context}"
        ))
        
        # Save to memory
        st.session_state.memory.save_context(
            {"input": input_text},
            {"output": response.content}
        )
        
        return response.content
        
    except Exception as e:
        return f"Error processing request: {str(e)}"

def main():
    st.title("🤖 File Analysis Agent")
    
    # Sidebar for file upload
    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader(
            "Choose files to analyze",
            accept_multiple_files=True
        )
        
        if uploaded_files:
            temp_dir, saved_files = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_files = saved_files
            st.success(f"Uploaded {len(uploaded_files)} files")
            
            # Display uploaded files
            st.subheader("Uploaded Files:")
            for file_name in saved_files.keys():
                st.write(f"- {file_name}")
            
            # Initial analysis
            if 'initial_analysis' not in st.session_state:
                st.session_state.initial_analysis = analyze_uploaded_files(saved_files)
                st.info("Initial file analysis completed. You can now ask questions about the files.")
        
        # Add a clear chat history button
        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.success("Chat history cleared!")

    # Main chat interface
    st.header("Chat with the Agent")
    
    # Initialize model and agent
    model = initialize_model()
    if model:
        model, tools, prompt = create_agent(model)
        
        # Chat input
        user_input = st.chat_input("Ask about your files or request changes...")
        
        if user_input:
            # Add user message to chat
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            
            # Get agent response
            try:
                # Process with agent
                response = process_with_agent(model, tools, prompt, user_input)
                
                # Add agent response to chat
                st.session_state.chat_history.append({
                    "role": "assistant",
                    "content": response
                })
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")
                st.error("Please try rephrasing your question or uploading the files again.")
        
        # Display chat history
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

if __name__ == "__main__":
    main() 