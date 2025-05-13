import streamlit as st
import os
import subprocess
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import tool
from langchain.memory import ConversationBufferMemory
from dotenv import load_dotenv
import tempfile
import shutil

# Load environment variables
load_dotenv()

# Initialize session state
if 'uploaded_files' not in st.session_state:
    st.session_state.uploaded_files = {}
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'memory' not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

st.set_page_config(page_title="File Analysis Agent", page_icon="🤖", layout="wide")

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

# File tools
@tool
def read_file(file_path: str) -> str:
    """Read and return the contents of a file.
    
    Args:
        file_path (str): Path to the file to read
        
    Returns:
        str: Contents of the file or error message if reading fails
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"Error reading file: {str(e)}"

@tool
def write_file(file_path: str, content: str) -> str:
    """Write content to a file.
    
    Args:
        file_path (str): Path where the file should be written
        content (str): Content to write to the file
        
    Returns:
        str: Success message or error message if writing fails
    """
    try:
        with open(file_path, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"Successfully wrote to {file_path}"
    except Exception as e:
        return f"Error writing to file: {str(e)}"

@tool
def list_files(directory: str) -> str:
    """List all files in a directory.
    
    Args:
        directory (str): Path to the directory to list
        
    Returns:
        str: Newline-separated list of files or error message if listing fails
    """
    try:
        files = os.listdir(directory)
        return "\n".join(files)
    except Exception as e:
        return f"Error listing files: {str(e)}"

@tool
def execute_python(code: str) -> str:
    """Execute Python code in a safe environment.
    
    Args:
        code (str): Python code to execute
        
    Returns:
        str: Execution result or error message if execution fails
    """
    try:
        local_vars = {}
        exec(code, {}, local_vars)
        return f"Execution successful. Output: {local_vars}"
    except Exception as e:
        return f"Execution failed: {str(e)}"

@tool
def analyze_dependencies(directory: str) -> str:
    """Analyze Python dependencies in a directory.
    
    Args:
        directory (str): Path to the directory to analyze
        
    Returns:
        str: Newline-separated list of imports found in Python files
    """
    try:
        imports = set()
        for file in os.listdir(directory):
            if file.endswith('.py'):
                with open(os.path.join(directory, file), 'r', encoding='utf-8') as f:
                    for line in f:
                        if line.startswith('import') or line.startswith('from'):
                            imports.add(line.strip())
        return "\n".join(imports)
    except Exception as e:
        return f"Error analyzing dependencies: {str(e)}"

@tool
def code_quality_check(file_path: str) -> str:
    """Run pylint code quality check on a Python file.
    
    Args:
        file_path (str): Path to the Python file to check
        
    Returns:
        str: Pylint output or error message if check fails
    """
    try:
        result = subprocess.run(["pylint", file_path], capture_output=True, text=True)
        return result.stdout
    except Exception as e:
        return f"Error running pylint: {str(e)}"

def create_agent(model):
    tools = [read_file, write_file, list_files, execute_python, analyze_dependencies, code_quality_check]

    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are an AI assistant that helps analyze and interact with files and folders.
You can read files, write files, list contents, execute Python code, analyze dependencies, and check code quality.
Always follow best practices, and provide clear and actionable responses.
"""),
        ("human", "{input}")
    ])

    return model, tools, prompt

def analyze_uploaded_files(saved_files):
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
    try:
        file_list = list_files(os.path.dirname(next(iter(st.session_state.uploaded_files.values()))))
        file_contents = {}
        for file_name, file_path in st.session_state.uploaded_files.items():
            content = read_file(file_path)
            file_contents[file_name] = content

        context = f"""Available files:
{file_list}

File contents:
{file_contents}

User question: {input_text}"""

        chat_history = st.session_state.memory.load_memory_variables({})["chat_history"]
        formatted_history = "\n".join([
            f"{'User' if msg.type == 'human' else 'Assistant'}: {msg.content}"
            for msg in chat_history
        ])

        response = model.invoke(prompt.format(
            input=f"{formatted_history}\n\n{context}"
        ))

        st.session_state.memory.save_context(
            {"input": input_text},
            {"output": response.content}
        )

        return response.content

    except Exception as e:
        return f"Error processing request: {str(e)}"

def main():
    st.title("🤖 File Analysis Agent")

    with st.sidebar:
        st.header("Upload Files")
        uploaded_files = st.file_uploader("Choose files to analyze", accept_multiple_files=True)
        if uploaded_files:
            temp_dir, saved_files = save_uploaded_files(uploaded_files)
            st.session_state.uploaded_files = saved_files
            st.success(f"Uploaded {len(uploaded_files)} files")
            st.subheader("Uploaded Files:")
            for file_name in saved_files.keys():
                st.write(f"- {file_name}")
            if 'initial_analysis' not in st.session_state:
                st.session_state.initial_analysis = analyze_uploaded_files(saved_files)
                st.info("Initial file analysis completed. You can now ask questions about the files.")

        if st.button("Clear Chat History"):
            st.session_state.chat_history = []
            st.session_state.memory.clear()
            st.success("Chat history cleared!")

    st.header("Chat with the Agent")
    model = initialize_model()
    if model:
        model, tools, prompt = create_agent(model)
        user_input = st.chat_input("Ask about your files or request advanced tasks...")

        if user_input:
            st.session_state.chat_history.append({"role": "user", "content": user_input})
            try:
                response = process_with_agent(model, tools, prompt, user_input)
                st.session_state.chat_history.append({"role": "assistant", "content": response})
            except Exception as e:
                st.error(f"Error getting response: {str(e)}")

        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])

if __name__== "__main__":
    main()