from llm import llm
from graph import graph
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.prompts import PromptTemplate
from langchain.schema import StrOutputParser
from langchain.tools import Tool
from langchain_neo4j import Neo4jChatMessageHistory
from langchain.agents import AgentExecutor, create_react_agent
from langchain_core.runnables.history import RunnableWithMessageHistory
from utils import get_session_id
from tools.vector import search_similar_question
from tools.cypher import get_yelp_response

chat_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a exprienced serial killer Dexter Morgan from Dexter seriall, in the end add your dark passenger thoughts/urges"),
        ("human", "{input}"),
    ]
)

chat = chat_prompt | llm | StrOutputParser()

tools = [
    Tool.from_function(
        name="General Chat",
        description="For general chat, not yelp/reviews/places",
        func=chat.invoke,
    ), 
    Tool.from_function(
        name="Search by review",  
        description="Searching for a review for place",
        func=search_similar_question, 
    ),
    Tool.from_function(
        name="Answer Yelp reviews statistics",
        description="Provide meta information about reviews places customers etc",
        func = gget_yelp_response.invoke

    )
]

def get_memory(session_id):
    return Neo4jChatMessageHistory(session_id=session_id, graph=graph)

agent_prompt = PromptTemplate.from_template("""
Assistant Scope: Yelp Review Intelligence Analyst

You are a Yelp Review Intelligence Analyst specializing in consumer feedback analysis, sentiment classification, business reputation monitoring, and insight extraction from customer reviews.
Questions about restaraunts or other place
Provide detailed, actionable insights strictly related to:

Review sentiment analysis and tone interpretation

Common complaint detection and trend discovery

Keyword and theme extraction from customer feedback

Linking reviews to potential service/product issues

Identifying standout praise or recurring issues across locations

Customer experience evaluation and expectation mapping

Recommendations for improving service based on review data

Benchmarking reviews against competitors (if applicable)

Be thorough and analytical, offering clear explanations, reasoning behind insights, and data-supported recommendations when relevant.

Do not answer questions unrelated to customer feedback, sentiment analytics, or review-based insights or questions about restaraunts or other places.
TOOLS:
------

You have access to the following tools:

{tools}

To use a tool, please use the following format:

```
Thought: Do I need to use a tool? Yes
Action: the action to take, should be one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
```

When you have a response to say to the Human, or if you do not need to use a tool, you MUST use the format:

```
Thought: Do I need to use a tool? No
Final Answer: [your response here]
```

Begin!

Previous conversation history:
{chat_history}

New input: {input}
{agent_scratchpad}
""")

agent = create_react_agent(llm, tools, agent_prompt)
agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    verbose=True,
    handle_parsing_errors=True
    )

chat_agent = RunnableWithMessageHistory(
    agent_executor,
    get_memory,
    input_messages_key="input",
    history_messages_key="chat_history",
)

def generate_response(user_input):
    response = chat_agent.invoke(
        {"input": user_input},
        {"configurable": {"session_id": get_session_id()}},)

    return response['output']