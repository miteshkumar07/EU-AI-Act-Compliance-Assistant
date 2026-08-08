from framework_pipeline.retrieval import build_ensemble_retriever
from langgraph.graph import StateGraph, START, END
from langchain_google_vertexai import ChatVertexAI
from typing import TypedDict, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from pydantic import BaseModel, Field


class AgentState(TypedDict):
    "Represent the state of the agent"
    "This is the memory of the agent, which is passed between nodes in the state graph."
    question: str
    chat_history: str
    documents: List[Document]
    generated_answer: str
    retries: int

retriever = build_ensemble_retriever()

# Breaks the user's query into 1-3 distinct search queries if the question is complex, or just 1 query if the question is simple.
class SearchQueries(BaseModel):
    queries: List[str] = Field(description="A list of 1 to 3 distinct search queries derived from the user's question.")

decomposer_llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
structured_decomposer = decomposer_llm.with_structured_output(SearchQueries)

decomposer_prompt = ChatPromptTemplate.from_template("""
You are an expert researcher. The user has asked a complex question about the EU AI Act.
Your job is to break this question down into 1 to 3 targeted, specific search queries that will be sent to a vector database.
If the question is simple, just return 1 query. If it asks to compare two things, return 2 queries.
User Question: {question}
""")
query_decomposer = decomposer_prompt | structured_decomposer


# Determines the intent of the user's question
class IntentDecision(BaseModel):
    decision: str = Field(description="Return 'greeting' if casual conversation, 'legal' if asking about regulations or AI Act.")

intent_llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
structured_intent = intent_llm.with_structured_output(IntentDecision)

def route_intent(state: AgentState):
    print("\n---NODE: INTENT CLASSIFIER---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template("""
    Classify the intent of the following user message: {question}
    
    Recent Chat History:
    {chat_history}
    
    Is it a casual greeting, a general conversational follow-up (e.g. "hi", "how are you", "summarize the last message"), or does it require searching the EU AI Act vector database (e.g. "what are penalties", "explain article 5", "ai in hiring")?
    Return 'greeting' if it's casual or general chat, and 'legal' if it requires searching the documents.
    """)
    decision_obj = (prompt | structured_intent).invoke({"question": question, "chat_history": state.get("chat_history", "")})
    decision = decision_obj.decision.lower()
    
    if decision == "greeting":
        print("---DECISION: GENERAL CHAT. ROUTING TO CHAT BYPASS---")
        return "greeting"
    else:
        print("---DECISION: LEGAL QUESTION. ROUTING TO RETRIEVER---")
        return "legal"

# Handles casual greetings
def chat_node(state: AgentState):
    print("\n---NODE: CASUAL CHAT BYPASS---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template("""
    You are an expert legal AI assistant specializing in the EU AI Act. The user just said: {question}
    
    Recent Chat History:
    {chat_history}
    
    Respond politely and helpfully. If they are asking you to summarize or reference previous messages, do so using the chat history provided. If it's just a greeting, introduce yourself and ask how you can help with the EU AI Act today.
    """)
    chain = prompt | llm | StrOutputParser()
    response = chain.invoke({"question": question, "chat_history": state.get("chat_history", "")})
    # Since we didn't search, we ensure documents is an empty list
    return {"generated_answer": response, "documents": []}

# Handles the retrieval of documents from the vector database
def retriever_node(state: AgentState):
    print("\n---NODE: RETRIEVING DOCUMENTS (MULTI-HOP)---")
    original_question = state["question"]
    
    # 1. Decompose the question into smaller searches
    search_queries_obj = query_decomposer.invoke({"question": original_question})
    queries = search_queries_obj.queries
    print(f"Decomposed into {len(queries)} queries: {queries}")
    
    all_documents = []
    
    # 2. Run a search for every single query!
    for q in queries:
        print(f"Searching for: {q}")
        docs = retriever.invoke(q)
        all_documents.extend(docs)
        
    # 3. Deduplicate the documents
    unique_documents = []
    seen_content = set()
    for doc in all_documents:
        if doc.page_content not in seen_content:
            seen_content.add(doc.page_content)
            unique_documents.append(doc)
            
    print(f"Total unique chunks retrieved: {len(unique_documents)}")
    return {"documents": unique_documents}


# Finally generates the answer based on the retrieved documents and the user's question
llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
prompt = ChatPromptTemplate.from_template("""
You are an expert legal AI assistant specializing in the European Union Artificial Intelligence Act (EU AI Act).
Your goal is to provide highly accurate, professional, and helpful answers based ONLY on the retrieved legal context.

Context from the EU AI Act:
{context}

User's Question: {question}

Answer the question clearly and thoroughly. If the context does not contain the answer, politely state that you do not know based on the provided documents.
CRITICAL: You MUST include inline citations for every factual claim you make. Use the exact [Doc N] [Source: X | Section: Y] metadata tags provided in the context blocks above to cite your claims (e.g. "The penalty is up to 35M EUR [Doc 3] [Source: clean_eu_ai_act.md | Section: Article 5]").
""")
chain = prompt | llm | StrOutputParser()

def format_docs_with_metadata(docs):
    formatted = []
    for i, d in enumerate(docs):
        source = d.metadata.get('source', 'Unknown Source')
        section = d.metadata.get('section', 'Unknown Section')
        formatted.append(f"[Doc {i+1}] [Source: {source} | Section: {section}]\n{d.page_content}")
    return "\n\n---\n\n".join(formatted)

def generation_node(state: AgentState):
    print("\n---NODE: GENERATING ANSWER (WITH CITATIONS)---")
    
    # Initialize or increment retries
    current_retries = state.get("retries", 0)
    
    question = state["question"]
    documents = state["documents"]
    context = format_docs_with_metadata(documents)
    generation = chain.invoke({"context": context, "question": question})
    
    return {"generated_answer": generation, "retries": current_retries + 1}



# Grades the relevance of retrieved documents using gemini.
class GradeDocument(BaseModel):
    """ Binary score for the retrieved document based on the relevance to the question. """
    binary_score: str = Field(description="Binary score 'yes' or 'no' indicating relevance of the document to the question.")
grader_llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
structured_llm_grader = grader_llm.with_structured_output(GradeDocument)
grader_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing relevance. If the document contains keywords or
  semantic meaning related to the question, grade it as relevant. Give a binary score 'yes' or 'no'.
Context from the EU AI Act:
{context}

User's Question: {question}

""")
retrieval_grader = grader_prompt | structured_llm_grader

# Checks if the generated answer is grounded in the retrieved documents
class GradeHallucinations(BaseModel):
    """Binary score for hallucination present in generation answer."""
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

structured_hallucination_grader = grader_llm.with_structured_output(GradeHallucinations)
hallucination_prompt = ChatPromptTemplate.from_template("""
You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts.
Give a binary score 'yes' or 'no'. 'Yes' means that the answer is grounded in and supported by the set of facts.

Set of facts:
{documents}

LLM generation: {generation}
""")

# Hallucination Checker
hallucination_grader = hallucination_prompt | structured_hallucination_grader

def check_hallucination(state: AgentState):
    print("\n---NODE: HALLUCINATION CHECKER---")
    question = state["question"]
    documents = state["documents"]
    generation = state["generated_answer"]
    retries = state.get("retries", 0)
    
    docs_str = format_docs_with_metadata(documents)
    score = hallucination_grader.invoke({"documents": docs_str, "generation": generation})
    
    if score.binary_score.lower() == "yes":
        print("---DECISION: GENERATION IS GROUNDED IN DOCUMENTS. ROUTING TO END---")
        return "useful"
    else:
        if retries >= 3:
            print(f"---DECISION: HALLUCINATION DETECTED {retries} TIMES. FORCING EXIT---")
            return "useful"
        
        print(f"---DECISION: HALLUCINATION DETECTED (Retry {retries}/3). RE-GENERATING---")
        return "not supported"



# If no relevant documents are found, we can try to rewrite the question to be more specific.
def rewrite_question(state: AgentState):
    """ If the filter documents node returns no documents, we can try to rewrite the question to be more specific."""
    print("\n---NODE: REWRITING QUESTION---")
    llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
    prompt = ChatPromptTemplate.from_template("""You are the best question rewritter that optimizes questions for vector search. 
Look at the input and try to reason about the underlying semantic intent / meaning in the context of the EU AI Act.
So, rewrite the following question: {question}""")
    chain = prompt | llm | StrOutputParser()
    better_question = chain.invoke({"question": state["question"]})
    print(f"Original: {state['question']} \nRewritten: {better_question}")
    return {"question": better_question}

# If no relevant document and the question is vague, we can ask the user to clarify their question.
def clarification_node(state: AgentState):
    print("\n---NODE: GENERATING CLARIFICATION REQUEST---")
    question = state["question"]
    prompt = ChatPromptTemplate.from_template("""
    The user asked: {question}
    You searched the EU AI Act but could not find a specific answer because the question is too vague or broad. 
    Write a polite, 1-2 sentence response asking the user to clarify their question. Give them 2 examples of what they might be looking for in the EU AI Act.
    """)
    chain = prompt | llm | StrOutputParser()
    clarification_response = chain.invoke({"question": question})
    return {"generated_answer": clarification_response}

class RoutingDecision(BaseModel):
    decision: str = Field(description="Return 'clarify' if vague, 'rewrite' if specific.")

routing_llm = ChatVertexAI(model_name="gemini-2.5-flash", temperature=0, project="ragbench-aiact")
structured_routing = routing_llm.with_structured_output(RoutingDecision)

def decide_to_generate(state: AgentState):
    print("\n---NODE: DECISION MAKER---")
    if len(state["documents"]) == 0:
        print("---DECISION: NO RELEVANT DOCUMENTS FOUND. EVALUATING QUESTION VAGUENESS---")
        question = state["question"]
        prompt = ChatPromptTemplate.from_template("""
        The user asked: {question}
        We found no relevant documents in the EU AI Act. Is this question too vague to search for (e.g. 'What is the penalty?', 'Tell me about AI'), or is it specific enough that it could just be phrased poorly and needs rewriting for vector search (e.g. 'Are there fines for using AI in hiring')?
        If it's too vague, return 'clarify'. If it's specific, return 'rewrite'.
        """)
        chain = prompt | structured_routing
        decision_obj = chain.invoke({"question": question})
        decision = decision_obj.decision.lower()
        
        if decision == "clarify":
            print("---DECISION: QUESTION IS VAGUE. ROUTING TO CLARIFY---")
            return "clarify"
        else:
            print("---DECISION: QUESTION IS SPECIFIC. ROUTING TO REWRITE---")
            return "rewrite"
    else:
        print("---DECISION: FOUND RELEVANT DOCUMENTS. ROUTING TO GENERATE---")
        return "generate"


# Defining all the nodes and edges in the state graph for the agent's workflow
workflow = StateGraph(AgentState)
workflow.add_node("chat_node", chat_node)
workflow.add_node("retriever", retriever_node)
workflow.add_node("generated_answer", generation_node)
workflow.add_node("rewrite_question", rewrite_question)
workflow.add_node("clarification_node", clarification_node)

workflow.add_conditional_edges(
    START,
    route_intent,
    {
        "greeting": "chat_node",
        "legal": "retriever"
    }
)
workflow.add_edge("chat_node", END)
workflow.add_conditional_edges(
    "retriever",
    decide_to_generate,
    {
        "generate": "generated_answer",
        "rewrite": "rewrite_question",
        "clarify": "clarification_node"
    }
)
workflow.add_edge("rewrite_question", "retriever")
workflow.add_edge("clarification_node", END)

workflow.add_conditional_edges(
    "generated_answer",
    check_hallucination,
    {
        "useful": END,
        "not supported": "generated_answer"
    }
)

app = workflow.compile()


if __name__ == "__main__":
    png_data = app.get_graph().draw_mermaid_png()
    with open("langgraph_workflow.png", "wb") as f:
        f.write(png_data)
        
    # Test question
    inputs = {"question": "What is the difference in penalties between using prohibited AI vs failing data governance obligations?"}
        
    # Stream the steps as the agent runs
    for output in app.stream(inputs):
        for key, value in output.items():
            print(f"Finished node: {key}")
            
    # Print the final answer!
    print(value["generated_answer"])