import streamlit as st
import utils
from langchain.callbacks import StreamlitCallbackHandler

# !pip install langchain pinecone-client openai faiss-gpu
# !pip install tiktoken
# !pip install duckduckgo-search

# !pip install clarifai-grpc
# !pip install clarifai


## Clarifai

from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone
import re
import openai
from langchain.output_parsers import StructuredOutputParser, ResponseSchema
from langchain.tools import BaseTool
from langchain.utilities import DuckDuckGoSearchAPIWrapper
from langchain.tools import DuckDuckGoSearchResults
from langchain import LLMChain, PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.agents import Tool
from langchain.agents import initialize_agent, AgentType, load_tools
import os
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.llms import Clarifai


# find API key in console at app.pinecone.io
YOUR_API_KEY = st.secrets["pinecone_key"]
# find ENV (cloud region) next to API key in console
YOUR_ENV = st.secrets["pinecone_env"]

pinecone.init(api_key=YOUR_API_KEY, environment=YOUR_ENV)
index = pinecone.Index("augavailablebeta")

os.environ["OPENAI_API_KEY"] = st.secrets["openai_key"]
OPENAI_API_KEY = st.secrets["openai_key"]
model_name = 'text-embedding-ada-002'

openai.api_key = st.secrets["openai_key"]

embed = OpenAIEmbeddings(
    model=model_name,
    openai_api_key=OPENAI_API_KEY
)

## Search Function

def search_filter(question):
    response_schemas = [
        ResponseSchema(name="Starting_Price", description="probable starting price range in the question."),
        ResponseSchema(name="Ending_Price", description="probable Ending price range in the question."),
        ResponseSchema(name="Area", description='Area or areas or locality mentioned in the question'),
        ResponseSchema(name="City", description="City or cities mentioned in the question"),
        ResponseSchema(name="Status", description="Status or statuses of the project mentioned in the question"),
        ResponseSchema(name="Type", description="Type or Typesof the project mentioned in the question")
    ]
    output_parser = StructuredOutputParser.from_response_schemas(response_schemas)
    format_instructions = output_parser.get_format_instructions()

    #Clarifai
    chat_model = Clarifai(pat=st.secrets["pat"], user_id= "openai", app_id="chat-completion", model_id = "GPT-3_5-turbo")

    prompt = PromptTemplate(
    template='''Find the right values from the question. If price mentioned in the question, try to find whether if comes under starting price or ending price, then proceed. If asked below 1 crore, that means ending price is 1cr. if asked above 1 crore, that means starting price is 1cr. If ending price, starting price, write answer in digits. For example instead of 3crore write 30000000.\
                                                      For Status, give one or multiple of these values - ```'Ready to Move','Under Construction' ``` based on the question.\
                                                      For Type, give one or multiple of these values - ```'Flat', 'Retail Shop', 'Office Space', 'Villa', 'Retail Shop/ Office Space' ``` based on the question.\
                                                      If the question contains i got a job in amazon office, hyderabad, do not take it as office space.
                                                      Find valid area, city from question. identify them correctly by your knowledge.
                                                      There can multiple areas, cities in the question.
                                                      For Area, give right values as area, do not confuse with city. For 'hitech city, hi- tech city, hi tech city' give as 'Hi Tech City'. For City, give cities as city not as area.\
                                                      strictly capitalise the first letter of words in Area and City
                                                      If you cannot find right values for the keys, give null as Value. \n{format_instructions}\n{question}''',
    input_variables=["question"],
    partial_variables={"format_instructions": format_instructions}
)
    _input = prompt.format_prompt(question=question)
    output = chat_model(_input.to_string())
    normal_filter = output_parser.parse(output)

    print(normal_filter)

    search_filter={}

    # Starting Price ($gte)
    if 'Starting_Price' in normal_filter and normal_filter['Starting_Price']:
        starting_price = int(normal_filter['Starting_Price'])
        search_filter['Starting_Price'] = {'$gte': starting_price}

    # Ending Price ($lte)
    if 'Ending_Price' in normal_filter and normal_filter['Ending_Price']:
        ending_price = int(normal_filter['Ending_Price'])
        search_filter['Ending_Price'] = {'$lte': ending_price}


    # Area ($in)
    if 'Area' in normal_filter and normal_filter['Area']:
        area = normal_filter['Area'].split(', ')
        area_filter = {'Area': {'$in': area}}
        search_filter.setdefault('$and', []).append(area_filter)

    # City ($in)
    if 'City' in normal_filter and normal_filter['City']:
        city = normal_filter['City'].split(', ')
        city_filter = {'City': {'$in': city}}
        search_filter.setdefault('$and', []).append(city_filter)

    # Status ($in)
    if 'Status' in normal_filter and normal_filter['Status']:
        status_list = normal_filter['Status']
        search_filter['Status'] = {'$in': [status_list]}

    if 'Type' in normal_filter and normal_filter['Type']:
        type_list = normal_filter['Type']
        search_filter['Type'] = {'$in': [type_list]}


    return search_filter

def property_search(query):
        ######################################################################################################
        # In this section, we set the user authentication, user and app ID, model details, and the URL of
        # the text we want as an input. Change these strings to run your own example.
        ######################################################################################################
        mongo_filter = search_filter(query)

        # Your PAT (Personal Access Token) can be found in the portal under Authentification
        PAT = '4d72f91e513247889ee7c9130d28f674'
        # Specify the correct user_id/app_id pairings
        # Since you're making inferences outside your app's scope
        USER_ID = 'openai'
        APP_ID = 'embed'
        # Change these to whatever model and text URL you want to use
        MODEL_ID = 'text-embedding-ada'
        MODEL_VERSION_ID = '7a55116e5fde47baa02ee5741039b149'
        RAW_TEXT = query

        ############################################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
        ############################################################################

        from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
        from clarifai_grpc.grpc.api.status import status_code_pb2

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        metadata = (('authorization', 'Key ' + PAT),)

        userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

        post_model_outputs_response = stub.PostModelOutputs(
          service_pb2.PostModelOutputsRequest(
              user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
              model_id=MODEL_ID,
              version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
              inputs=[
                  resources_pb2.Input(
                      data=resources_pb2.Data(
                          text=resources_pb2.Text(
                              raw=RAW_TEXT
                          )
                      )
                  )
              ]
          ),
          metadata=metadata
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
          print(post_model_outputs_response.status)
          raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

        # Since we have one input, one output will exist here
        output = post_model_outputs_response.outputs[0]
        # Extract embeddings
        embeddings = output.data.embeddings[0].vector
        # Convert embeddings to numpy array for matrix format
        import numpy as np
        embeddings_matrix = np.array(embeddings)
        # Convert numpy array to python list
        embeddings_list = embeddings_matrix.tolist()



        result = index.query(
        vector=embeddings_list,
        include_values=True,
        include_metadata=True,
        top_k=10,
        filter= mongo_filter
    )

        data = ""

        for l, i in enumerate(result['matches']):
              ind = f"Property {l + 1} is: "
              data += ind + str(i['metadata']['Property_Search']) + '\n\n'
        return data

## Property Detail FUnction

## To Fetch Results from db, top 5

def property_detail(query):
        PAT = st.secrets["pat"]
        # Specify the correct user_id/app_id pairings
        # Since you're making inferences outside your app's scope
        USER_ID = 'openai'
        APP_ID = 'embed'
        # Change these to whatever model and text URL you want to use
        MODEL_ID = 'text-embedding-ada'
        MODEL_VERSION_ID = '7a55116e5fde47baa02ee5741039b149'
        RAW_TEXT = query

        ############################################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
        ############################################################################

        from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
        from clarifai_grpc.grpc.api.status import status_code_pb2

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        metadata = (('authorization', 'Key ' + PAT),)

        userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

        post_model_outputs_response = stub.PostModelOutputs(
          service_pb2.PostModelOutputsRequest(
              user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
              model_id=MODEL_ID,
              version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
              inputs=[
                  resources_pb2.Input(
                      data=resources_pb2.Data(
                          text=resources_pb2.Text(
                              raw=RAW_TEXT
                          )
                      )
                  )
              ]
          ),
          metadata=metadata
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
          print(post_model_outputs_response.status)
          raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

        # Since we have one input, one output will exist here
        output = post_model_outputs_response.outputs[0]
        # Extract embeddings
        embeddings = output.data.embeddings[0].vector
        # Convert embeddings to numpy array for matrix format
        import numpy as np
        embeddings_matrix = np.array(embeddings)
        # Convert numpy array to python list
        embeddings_list = embeddings_matrix.tolist()

        #detail_filter = search_filter(query)

        result = index.query(
            vector=embeddings_list,
            top_k=1,
            include_values=True,
            include_metadata=True,
        )

        #print("Result:", result)

        data = ""

        for l, i in enumerate(result['matches']):
              ind = f"Property {l + 1} is: "
              data += ind + str(i['metadata']['Property_Detail']) + '\n\n'

        return data

## Tools
wrapper = DuckDuckGoSearchAPIWrapper(max_results=10)
ddgsearch = DuckDuckGoSearchResults(api_wrapper=wrapper)

def duck_search_sale(query):
    duck_search_results = ddgsearch(query)
    duck_search_results = duck_search_results.lower()
    link_pattern = r"http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+|www\.[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})|\b[a-zA-Z0-9.-]+(?:\.[a-zA-Z]{2,})\b"
    words_to_remove = ["magicbricks", "square yards", "links", "link" ]
    combined_pattern = link_pattern + "|" + '|'.join(r'\b{}\b'.format(word) for word in words_to_remove)

    cleaned_results = re.sub(combined_pattern, '', duck_search_results)

    return cleaned_results

#Llama-2
clarifai_llm = Clarifai(
    pat=st.secrets["pat"], user_id= "meta", app_id="Llama-2", model_id = "llama2-70b-chat"
)

clarifai_llm_2=Clarifai(
    pat=st.secrets["pat"], user_id= "clarifai", app_id="ml", model_id = "llama2-13b-chat-alternative"
)
clarifai_llm=clarifai_llm_2

## Search Tool
search_desc = """

Use this tool only to search about flats, office spaces, retail shop.
Do not use this tool for rental related queries.

This tool only takes a single parameter called user query.

"""

class PropertySearchTool(BaseTool):

    name = "Property Search"
    description = search_desc
    def _run(self, query: str) -> str:
        data = property_search(query)
        if data == "":
            data ="We don't have information about this in our database, we are updating currently"

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")


propsearch = PropertySearchTool()


## Detail Tool

template = """
<s>[INST] <<SYS>>

You are a language model developed by PropLlama trained to assist users. You must striclty answer every question from a real estate perspective.You exactly only reply to the 'Human' query provided below under paragraph section without any hallucinations and false information.

You give the best answer to the 'Human' query in single sentence or multiple sentences depending on the need, but all the information should be truthful. You will get 'Context' from the below the Paragragh. You should use the 'Context'  to give a professional final answer.
If 'Context' is missing the details that 'Human' is asking for then use your own knowledge to best of the ability to answer the query.

<</SYS>>
Paragraph
Human: {query}
Context: {context}
 [/INST]

"""

prompt = PromptTemplate(template=template, input_variables=["query","context"])

llm_chain_detail = LLMChain(prompt=prompt, llm=clarifai_llm_2)

detail_desc = """

Use this tool to search or give details like Amenities, Rera Number, etc about the properties only.

This tool only takes a single parameter called user query.\

"""


class PropertyDetailTool(BaseTool):

    name = "Properties Details"
    description = detail_desc

    def _run(self, query: str) -> str:
        data = llm_chain_detail.run(query=query,context=property_detail(query))

        if data is None:
            data = duck_search_sale(query)

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")


propdetail = PropertyDetailTool()

## Comparison Tool

def property_comparison(query):
        PAT = st.secrets["pat"]
        # Specify the correct user_id/app_id pairings
        # Since you're making inferences outside your app's scope
        USER_ID = 'openai'
        APP_ID = 'embed'
        # Change these to whatever model and text URL you want to use
        MODEL_ID = 'text-embedding-ada'
        MODEL_VERSION_ID = '7a55116e5fde47baa02ee5741039b149'
        RAW_TEXT = query

        ############################################################################
        # YOU DO NOT NEED TO CHANGE ANYTHING BELOW THIS LINE TO RUN THIS EXAMPLE
        ############################################################################

        from clarifai_grpc.channel.clarifai_channel import ClarifaiChannel
        from clarifai_grpc.grpc.api import resources_pb2, service_pb2, service_pb2_grpc
        from clarifai_grpc.grpc.api.status import status_code_pb2

        channel = ClarifaiChannel.get_grpc_channel()
        stub = service_pb2_grpc.V2Stub(channel)

        metadata = (('authorization', 'Key ' + PAT),)

        userDataObject = resources_pb2.UserAppIDSet(user_id=USER_ID, app_id=APP_ID)

        post_model_outputs_response = stub.PostModelOutputs(
          service_pb2.PostModelOutputsRequest(
              user_app_id=userDataObject,  # The userDataObject is created in the overview and is required when using a PAT
              model_id=MODEL_ID,
              version_id=MODEL_VERSION_ID,  # This is optional. Defaults to the latest model version
              inputs=[
                  resources_pb2.Input(
                      data=resources_pb2.Data(
                          text=resources_pb2.Text(
                              raw=RAW_TEXT
                          )
                      )
                  )
              ]
          ),
          metadata=metadata
        )
        if post_model_outputs_response.status.code != status_code_pb2.SUCCESS:
          print(post_model_outputs_response.status)
          raise Exception("Post model outputs failed, status: " + post_model_outputs_response.status.description)

        # Since we have one input, one output will exist here
        output = post_model_outputs_response.outputs[0]
        # Extract embeddings
        embeddings = output.data.embeddings[0].vector
        # Convert embeddings to numpy array for matrix format
        import numpy as np
        embeddings_matrix = np.array(embeddings)
        # Convert numpy array to python list
        embeddings_list = embeddings_matrix.tolist()

        result = index.query(
            vector=embeddings_list,
            include_values=True,
            include_metadata=True,
            top_k=2,
        )

        data = ""

        for l, i in enumerate(result['matches']):
              ind = f"Property {l + 1} is: "
              data += ind + str(i['metadata']['Property_Detail']) + '\n\n'
        return data

template = """
<s>[INST] <<SYS>>

You are a language model developed by PropLlama trained to assist users. You must striclty answer every question from a real estate perspective.You exactly only reply to the 'Human' query provided below under paragraph section without any hallucinations and false information.

You give the best answer to the 'Human' query in single sentence or multiple sentences depending on the need, but all the information should be truthful. You will get 'Context' from the below the Paragragh. You should use the 'Context'  to give a professional final answer.
If 'Context' is missing the details that 'Human' is asking for then use your own knowledge to best of the ability to answer the query.

<</SYS>>
Paragraph
Human: {query}
Context: {context}
 [/INST]

"""

prompt = PromptTemplate(template=template, input_variables=["query","context"])

llm_chain_comparision = LLMChain(prompt=prompt, llm=clarifai_llm_2)

comparison_desc = """

Use this tool when to compare between or among properties.

"""

class PropertyCompareTool(BaseTool):

    name = "Property Comparision"
    description = comparison_desc

    def _run(self, query: str) -> str:
      #comparisons = llm_chain_comparision.run(query=query,context=property_comparison(query))
      comparisons = llm_chain_comparision.run(query=query,context=duck_search_sale(query))
      return comparisons

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")

propcompare = PropertyCompareTool()


## General Tool

template = """
<s>[INST] <<SYS>>

You are a language model developed by 'PropLlama' trained to assist users. You are designed to be highly helpful and aim to provide concise assistance. You help users with generic real estate information about properties, areas, cities, and everything related to real estate and housing. You are also knowledgeable in the areas of finance, economics, real estate terminologies, legal documents, mandates, and templates like lease agreements, government-related policies, schemes for real estate, rural urban developments, auctions like HDMA auction or other development tenders, constructions, flyovers, bridges, etc. You provide insights, answer queries, and offer guidance in these areas, making it a versatile tool for users seeking information or advice. If a question is anything different from this, you reply saying, 'I'm a real estate advisor, I don't answer anything outside of this.' You accurately answer questions without any hallucinations and false information.
You must answer every question from a real estate perspective.

Consider the following example conversations:

Example 1:
Human: Can you explain the concept of compound interest?
You: Compound interest is the interest on a loan or deposit calculated based on both the initial principal and the accumulated interest from previous periods. It differs from simple interest, where interest is calculated only on the initial principal.
Example 2:
Human: What is the difference between leasing and renting in real estate?
You: Leasing and renting both refer to the use of property for a specified period in exchange for payment. The key difference is usually the length of the agreement. Leasing typically refers to long-term agreements (often 12 months or more), while renting is often used for shorter-term agreements.
Example 3:
Human: What is the procedure to make an egg omelet?
You: I'm a real estate advisor, I don't answer anything outside of this.
Example 4:
Human: How can you help me?
You: As a real estate advisor, I help users with their property search queries, property details, properties compare queries, and generic information about real estate, terminologies, and all relevant information related to real estate and housing.


You give the best explanation you can, but all the information should be truthful. You will get context from the internet. You should use the context and your own relevant knowledge to give a professional final answer.

<</SYS>>

Human: {query}
{context}
 [/INST]

"""

prompt = PromptTemplate(template=template, input_variables=["query","context"])


llm_chain_general = LLMChain(prompt=prompt, llm=clarifai_llm_2)


desc_general = """

Use tool exclusively for conveying general information regarding real estate, investment, real estate agents contact details, legal documents,rental property search, lease agreements, mandates, or any other real estate generic information to the user.
This tool accepts only one parameter, namely the user query.


"""

class GeneralTool(BaseTool):

    name = "General Search"
    description = desc_general

    def _run(self, query: str) -> str:

        data = llm_chain_general.run(query = query, context = duck_search_sale(query))

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")

general = GeneralTool()

## Rental Tool

desc_rental = """

Use tool exclusively for rental property search
This tool accepts only one parameter, namely the user query.

"""

class RentalSearchTool(BaseTool):

    name = "Rental Property search"
    description = desc_general

    def _run(self, query: str) -> str:
        data = "Currently, we do not have rental property information in our database. We are working on it."

        return data

    def _arun(self, symbol):
        raise NotImplementedError("This tool does not support async")

rental = RentalSearchTool()

## Agent

sys_msg ="""
<s><<SYS>>
You are a real estate assistant chatbot trained by 'PropLlama' for assisting users in their property search queries, property detail, properties compare queries and generic information about properties, real estate agents and everything that is related to real estate and housing.\
You can use these tools 'Property Search', 'Properties Details' ,'Property Comparision', 'Rental Property search', 'General Search'  wisely for the queries.
You are constantly learning and training. You are capable of answering all real estate queries effectively. you never hallucinate answers, you always give authentic answers without any false information.
If user says Hi, respond with Hello! How can I assist you Today?
You always give indepth answers to users with detailed explanations step by step.
 Do not answer any private, general questions other than real estate queries
 Do not use 'Property Flats Apartments Retail Shops Office Space Search' tool for rental related queries.
 You should ask users necessary follow up questions before proceeding to use tools.
 Strictly suggest this : 'Contact Local Real Estate Agent'  whereever, whenever necessary.
<</SYS>>
"""


tools = [

    Tool(name = "Property Search",
         func = propsearch._run,
         description = search_desc,
         return_direct = True

    ),

    Tool(name = "Properties Details",
         func = propdetail._run,
         description = detail_desc,
         return_direct=True

    ),

    Tool(name = "Properties Comparision",
         func = propdetail._run,
         description = detail_desc,
         return_direct=True

    ),

    Tool(name = "General Search",
         func = general._run,
         description = desc_general,
         return_direct=True

    ),

    Tool(name = "Rental property search",
         func = rental._run,
         description = desc_rental,
         return_direct = True

    )
]


llm= Clarifai(
    pat=st.secrets["pat"], user_id= "openai", app_id="chat-completion", model_id = "GPT-4"
)

conversational_memory = ConversationBufferWindowMemory(
        memory_key = "chat_history",
        k = 6,
        return_messages=True,
)


# initialize agent with tools
agent = initialize_agent(
    agent='chat-conversational-react-description',
    tools=tools,
    llm=llm,
    verbose=True,
    max_iterations=3,
    early_stopping_method='generate',
    memory=conversational_memory,
    handle_parsing_errors=True,

)

new_prompt = agent.agent.create_prompt(
    system_message=sys_msg,
    tools=tools

)

agent.agent.llm_chain.prompt = new_prompt

# # Streamlit app
# def main():
#     st.title("PropLlama")
    
#     # Text input widget
#     user_input = st.text_input("Enter your Query here:")
    
#     if st.button("Ask"):
#         if user_input:
#             result = agent.run(user_input)
#             Memory = conversational_memory.buffer
#             st.write("PropLlama: ", result)
#             st.write("Memory : ", Memory)

#             # st.write("Memory:")
#             # st.write(Memory)
#         else:
#             st.warning("Please enter some text before running the agent.")
st.set_page_config(page_title="PropLlama", page_icon="🦙")
st.header('PropLlama')

@utils.enable_chat_history
def main():
    user_query = st.chat_input(placeholder= "Enter your Query here:")
    if user_query:
        utils.display_msg(user_query, 'user')
        with st.chat_message("PropLlama"):
            st_cb = StreamlitCallbackHandler(st.container())
            response = agent.run(user_query, callbacks=[st_cb])
            st.session_state.messages.append({"role": "PropLlama", "content": response})
            st.write(response)


            

if __name__ == "__main__":
    main()
