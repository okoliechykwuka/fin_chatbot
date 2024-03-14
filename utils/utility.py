import streamlit as st
from langchain_community.chat_models import ChatOpenAI
import os
from langchain.prompts.prompt import PromptTemplate
import openai
import base64
from lida import Manager, TextGenerationConfig , llm  
import pandas as pd
import numpy as np
import io
import os

class ExcelLoader():
    def __init__(self, file):
        self.status = False
        self.name =  'ExcelLoader'
        self.file = file
        self.loader = pd.ExcelFile
        self.ext = ['xlsx']
    
    def load(self):
        from langchain.document_loaders.csv_loader import CSVLoader

        ssheet = self.loader(self.file)
        docs = []
        for i,sheet in enumerate(ssheet.sheet_names):
            df = ssheet.parse(sheet)
    
            # Create an in-memory file-like object
            temp_buffer = io.StringIO()

            # Write DataFrame to the buffer instead of writing to disk
            df.to_csv(temp_buffer, index=False)

            # Reset the buffer position to the beginning
            temp_buffer.seek(0)

            docs.append(temp_buffer)
        return docs
    
def load_csv(file):

    temp_buffer = io.StringIO()
    df = pd.read_csv(file)
    # Write DataFrame to the buffer instead of writing to disk
    df.to_csv(temp_buffer, index=False)
    temp_buffer.seek(0)

    return temp_buffer



def randomName():
    n = []
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())
    n.append(str(np.random.randint(1,9)))                                 
    n.append(str(np.random.randint(1,9)))   
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']))         
    n.append(np.random.choice(['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']).upper())  

    return ''.join(n)       



def generate_plot(data_path, prompt=None,api_key=None):
    
    lida = Manager(text_gen = llm(provider="openai", api_key=api_key)) 
    textgen_config = TextGenerationConfig(n=1, temperature=0.5,
                                          model="gpt-3.5-turbo-0125", use_cache=False)

    
    summary = lida.summarize(data_path, summary_method="default", textgen_config=textgen_config)  
    
    #textgen_config = TextGenerationConfig(n=1, temperature=0, use_cache=True)
    if prompt == None:
        goals = lida.goals(summary, n=1, textgen_config=textgen_config)

    else:
        persona = prompt
        print(f"This the Prompt Recieved : {prompt}")
        goals = lida.goals(summary, n=1, persona=persona, textgen_config=textgen_config)
        
    i = 0

    library = np.random.choice(["seaborn", 'plotly',])

    try:
        #library = "plotly"
        plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)
        if not plots:
            st.write("Using Matplotlib")
            library = "matplotlib"
            plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)


    except Exception as e:
        st.write("Using Matplotlib")
        library = "matplotlib"
        plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)
    if len(plots) == 0:
        print("Could not generate a plot from your prompt. The below chart can be helpful")
        plots = lida.visualize(summary=summary, goal=goals[i], textgen_config=textgen_config, library=library)
        
    
    fig = plots[0]
    caption = goals[0].rationale

    return caption, fig


def classify_prompt(input, api_key=None):
    """
    This function classifies a user input to determine if it is requesting a plot.
    It returns True if the input is requesting a plot, and False otherwise.
    """

    # Updated prompt with clearer instructions
    prompt = """
    Analyze the following user input and determine if it is requesting a plot of a story, data, or any other form of graphical representation.
    
    If the input explicitly asks for a plot, description of a plot, or any graphical representation, respond with "PLOT REQUESTED".
    If the input does not ask for a plot or graphical representation, respond with "NO PLOT REQUESTED".
    
    User Input: "{}"
    """

    model = ChatOpenAI(model_name="gpt-3.5-turbo-16k-0613")
    res = model.predict(prompt.format(input))

    # Checking for the specific responses
    if res.strip() == "PLOT REQUESTED":
        return True
    else:
        return False



def display(fig, rationale):
    import base64
    
    # Decode the base64-encoded data
    fig_data = base64.b64decode(fig.raster)
    container = st.container()
    with container:
        
        st.image(fig_data,output_format='auto')
        st.markdown("**Insight**")
        st.write(rationale)
        
        st.download_button(label="Download Image",
                           data=fig_data,
                           file_name=f"img{randomName()}.jpg",
                           mime="image/jpeg",)
        
def data_load(ext, file):
    try:
        os.mkdir('temps')
    except FileExistsError:
        pass
    path = os.path.join(os.getcwd(), r"temps")
    if ext == "csv": 
        rname = randomName() + ".csv"
        file_path = os.path.join(path, rname)
        print("THIS IS FILE PATH !!!!!!!!!",file_path)
        file.to_csv(file_path, index=False)

    elif ext in ["xls","xlsx","xlsm","xlsb"]:
        rname = randomName() + ".xlsx"
        file_path = os.path.join(path, rname)
        file.to_excel(file_path),
    else:
        pass
    return file_path


def extract_data(text):
    import re
    import io
    # Define the regex pattern to extract the CSV string
    pattern = r'<CSV>(.*?)<CSV>'
    match = re.search(pattern, text, re.DOTALL)

    if match:
        csv_string = match.group(1).strip()

        # Use io.StringIO to create a file-like object
        csv_file = io.StringIO(csv_string)

        # Read the CSV into a pandas DataFrame
        df = pd.read_csv(csv_file)

        # Create an in-memory file-like object
        buffer = io.StringIO()  # For text data, you can use io.BytesIO() for binary data

        # Save DataFrame to the buffer instead of writing to disk
        df.to_csv(buffer, index=False)

        # Display the DataFrame
        print(buffer.getvalue())
    else:
        print("No CSV string found in the text.")
        buffer.seek()
    return buffer


def create_lida_data(file_paths, file_name=None):
    
    assert isinstance(file_paths, list)
    lida_buffer = io.BytesIO()

    with pd.ExcelWriter(lida_buffer, engine='xlsxwriter') as writer:
        # Write each DataFrame to a different sheet
        for i, file in enumerate(file_paths):
            file_buffer = io.StringIO()
            file_buffer.write(file.getvalue())
            # Reset the buffer position to the beginning
            file_buffer.seek(0)
            sheet = f'Sheet{i}'
            df = pd.read_csv(file_buffer)
            df.to_excel(writer, sheet_name=sheet, index=False)
    lida_buffer.seek(0)

    # Return the file path
    #print(lida_buffer.getvalue())
    #print(type(lida_buffer))
    excel = pd.read_excel(lida_buffer)
    return excel #lida_buffer.getvalue()
            
