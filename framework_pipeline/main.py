from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List, Optional
from langchain_agent import app as agent_app

app = FastAPI(title="EU AI Act API")
    
# What the user sends us
class ChatRequest(BaseModel):
    question: str
    chat_history: Optional[str] = ""

# What we send back
class DocumentInfo(BaseModel):
    content: str
    source: str
    section: str

class ChatResponse(BaseModel):
    answer: str
    citations: List[DocumentInfo]


@app.post("/chat", response_model=ChatResponse)
def chat_endpoint(request: ChatRequest):
    try:
        final_state = {}
        # Our LangGraph stream is synchronous, so this runs safely in FastAPI's threadpool
        for output in agent_app.stream({
            "question": request.question,
            "chat_history": request.chat_history
        }):
            # Accumulate the state changes from the agent
            for key, value in output.items():
                for k, v in value.items():
                    final_state[k] = v
        
        generated_answer = final_state.get("generated_answer", "Error: No answer generated.")
        docs = final_state.get("documents", [])
        
        # Format citations to return in the JSON
        citations = [
            DocumentInfo(
                content=doc.page_content,
                source=doc.metadata.get("source", "Unknown"),
                section=doc.metadata.get("section", "Unknown")
            ) for doc in docs
        ]
            
        return ChatResponse(answer=generated_answer, citations=citations)
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))