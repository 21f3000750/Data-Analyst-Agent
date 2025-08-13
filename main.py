# main.py

import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse

# Import our agent
from agent import DataAnalystAgent

# Define the total request timeout in seconds (3 minutes = 180s)
REQUEST_TIMEOUT = 175

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses an LLM agent to analyze data.",
)

# Initialize our agent
agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(
    questions_txt: UploadFile = File(..., alias="questions.txt"),
    files: Optional[List[UploadFile]] = File(None, alias="files"),
):
    """
    The main API endpoint that receives data analysis tasks.
    """
    all_files = [questions_txt]
    if files:
        all_files.extend(files)

    try:
        question_content = (await questions_txt.read()).decode("utf-8")
        # Reset pointer in case the file needs to be read again by the agent
        await questions_txt.seek(0)

        # Run the agent logic with the specified timeout.
        result = await asyncio.wait_for(
            run_agent_async(question_content, all_files),
            timeout=REQUEST_TIMEOUT
        )

        if result.get("status") == "success":
            return JSONResponse(content=result.get("result"))
        else:
            # If the agent failed all its retries, return an error
            raise HTTPException(status_code=500, detail=result.get("message"))

    except asyncio.TimeoutError:
        # This block executes if the asyncio.wait_for call exceeds the timeout.
        raise HTTPException(
            status_code=504, # 504 Gateway Timeout
            detail=f"Request timed out after {REQUEST_TIMEOUT} seconds."
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"An unexpected error occurred: {str(e)}")

async def run_agent_async(question: str, files: List[UploadFile]):
    """
    A wrapper to run the synchronous agent code in an async-compatible way.
    FastAPI runs this in a separate thread pool to avoid blocking the server.
    """
    return agent.run(question, files)

@app.get("/")
def read_root():
    return {"status": "Data Analyst Agent is running"}


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=8000)