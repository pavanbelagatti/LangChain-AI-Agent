import os
from dotenv import load_dotenv
import gradio as gr
from langchain_openai import ChatOpenAI
from langchain.agents import create_openai_tools_agent, AgentExecutor
from langchain_core.tools import Tool
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain_community.tools.ddg_search import DuckDuckGoSearchRun
from datetime import datetime
import requests
import json

# Load environment variables
load_dotenv()

# Initialize the language model
llm = ChatOpenAI(
    model="gpt-4",
    temperature=0
)

# Define custom tools
def get_current_time() -> str:
    """Get the current date and time."""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def calculator(expression: str) -> str:
    """Evaluate a mathematical expression."""
    try:
        return str(eval(expression))
    except Exception as e:
        return f"Error calculating: {str(e)}"

def get_latest_news(query: str = "", category: str = "") -> str:
    """
    Get the latest news headlines.
    Parameters:
    - query: Search term for specific news (optional)
    - category: News category like business, entertainment, health, science, sports, technology (optional)
    """
    api_key = os.getenv("NEWSAPI_API_KEY")
    if not api_key:
        return "News API key not found. Please set NEWSAPI_API_KEY in your .env file."
    
    # Construct the API request
    url = "https://newsapi.org/v2/top-headlines"
    params = {
        "apiKey": api_key,
        "language": "en",
        "pageSize": 5  # Limit to 5 articles for readability
    }
    
    # Add optional parameters if provided
    if query:
        params["q"] = query
    if category and category.lower() in ["business", "entertainment", "general", "health", "science", "sports", "technology"]:
        params["category"] = category.lower()
    elif not query:  # Default to general news if no query or category
        params["category"] = "general"
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            if news_data["totalResults"] == 0:
                # Try an alternative approach with everything endpoint for location-based searches
                return get_location_news(query)
            
            # Format the results
            result = f"Latest News {f'on {query}' if query else ''} {f'in {category}' if category else ''}:\n\n"
            for i, article in enumerate(news_data["articles"], 1):
                result += f"{i}. {article['title']}\n"
                result += f"   Source: {article['source']['name']}\n"
                result += f"   Published: {article['publishedAt']}\n"
                result += f"   Summary: {article['description'] if article['description'] else 'No description available'}\n"
                result += f"   URL: {article['url']}\n\n"
            
            return result
        else:
            return f"Error fetching news: {response.status_code}"
    except Exception as e:
        return f"Error processing news request: {str(e)}"

def get_location_news(location: str) -> str:
    """
    Get news for a specific location using the everything endpoint.
    This is better for location-based searches.
    """
    api_key = os.getenv("NEWSAPI_API_KEY")
    if not api_key:
        return "News API key not found. Please set NEWSAPI_API_KEY in your .env file."
    
    # Use the everything endpoint which is better for location searches
    url = "https://newsapi.org/v2/everything"
    params = {
        "apiKey": api_key,
        "q": location,  # Search for the location name
        "sortBy": "publishedAt",  # Sort by most recent
        "language": "en",
        "pageSize": 5
    }
    
    try:
        response = requests.get(url, params=params)
        if response.status_code == 200:
            news_data = response.json()
            
            if news_data["totalResults"] == 0:
                return f"No news found for location: {location}. Try a different search term or check back later."
            
            # Format the results
            result = f"Latest News related to {location}:\n\n"
            for i, article in enumerate(news_data["articles"], 1):
                result += f"{i}. {article['title']}\n"
                result += f"   Source: {article['source']['name']}\n"
                result += f"   Published: {article['publishedAt']}\n"
                result += f"   Summary: {article['description'] if article['description'] else 'No description available'}\n"
                result += f"   URL: {article['url']}\n\n"
            
            return result
        else:
            return f"Error fetching location news: {response.status_code}"
    except Exception as e:
        return f"Error processing location news request: {str(e)}"

# Create search tool
duckduckgo_search = DuckDuckGoSearchRun()

# Define the tools
tools = [
    Tool(
        name="Search",
        func=duckduckgo_search.run,
        description="Useful for searching the web for current information."
    ),
    Tool(
        name="Calculator",
        func=calculator,
        description="Useful for performing mathematical calculations. Input should be a mathematical expression."
    ),
    Tool(
        name="CurrentTime",
        func=get_current_time,
        description="Get the current date and time. No input is needed."
    ),
    Tool(
        name="LatestNews",
        func=get_latest_news,
        description="Get the latest news headlines. You can specify a search query and/or category (business, entertainment, health, science, sports, technology)."
    ),
    Tool(
        name="LocationNews",
        func=get_location_news,
        description="Get news for a specific location or city. Input should be the name of the location (e.g., 'Mumbai', 'New York')."
    )
]

# Create the agent prompt
prompt = ChatPromptTemplate.from_messages([
    ("system", """You are an intelligent assistant that helps users with their questions.
    You have access to tools that can search the web, get the latest news, perform calculations, and get the current time.
    Use these tools to provide helpful and accurate responses.
    
    When asked about general news or news categories, use the LatestNews tool.
    When asked about news in a specific location or city, use the LocationNews tool.
    
    Always think step by step and explain your reasoning clearly.
    """),
    MessagesPlaceholder(variable_name="chat_history"),
    ("human", "{input}"),
    MessagesPlaceholder(variable_name="agent_scratchpad")
])

# Create the agent
agent = create_openai_tools_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

# Create the agent executor
agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True,
    max_iterations=5
)

# Initialize chat history
chat_history = []

# Function to process user input
def process_input(message):
    global chat_history
    # Run the agent
    response = agent_executor.invoke({
        "input": message,
        "chat_history": chat_history
    })
    # Update chat history
    chat_history.append(HumanMessage(content=message))
    chat_history.append(AIMessage(content=response["output"]))
    return response["output"]

# Create the Gradio interface
with gr.Blocks(title="AI Agent Dashboard") as demo:
    gr.Markdown("# 🤖 AI Agent Dashboard")
    gr.Markdown("Ask me anything! I can search the web, get the latest news, perform calculations, and more.")
    
    chatbot = gr.Chatbot(height=500)
    msg = gr.Textbox(label="Your question", placeholder="Ask me about the latest news, search the web, or do calculations...")
    clear = gr.Button("Clear conversation")
    
    def respond(message, chat_history):
        bot_message = process_input(message)
        chat_history.append((message, bot_message))
        return "", chat_history
    
    def clear_chat():
        global chat_history
        chat_history = []
        return None
    
    msg.submit(respond, [msg, chatbot], [msg, chatbot])
    clear.click(clear_chat, None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)