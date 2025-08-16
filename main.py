import asyncio
from typing import List, Optional
from fastapi import FastAPI, File, UploadFile, HTTPException,Request
from fastapi.responses import JSONResponse

from agent import DataAnalystAgent

REQUEST_TIMEOUT = 175

app = FastAPI(
    title="Data Analyst Agent API",
    description="An API that uses an LLM agent to analyze data.",
)

agent = DataAnalystAgent()

@app.post("/api/")
async def analyze_data(request: Request
):
    """
    The main API endpoint that receives data analysis tasks.
    """

    
    form = await request.form()
    questions_txt = None
    other_files = []

    print(f"Form fields received: {list(form.keys())}")

    for key, value in form.items():

        print(f"1Processing file field '{key}' with filename '{value.filename}'")
        print(type(value))
        print(isinstance(value, UploadFile))
        if key == "questions.txt":
                print(f"Found questions file in field '{key}' with filename '{value.filename}'")
                questions_txt = value
        else:
            print(f"Found data file in field '{key}' with filename '{value.filename}'")
            other_files.append(value)

    if not questions_txt:
        print("'questions.txt' field was not found in the form.")
        raise HTTPException(status_code=400, detail="questions.txt file is missing.")

    all_files = [questions_txt] + other_files


    try:
        question_content = (await questions_txt.read()).decode("utf-8")
        await questions_txt.seek(0)

        result = await asyncio.wait_for(
            run_agent_async(question_content, all_files),
            timeout=REQUEST_TIMEOUT
        )

        if result.get("status") == "success":
            return JSONResponse(content=result.get("result"))
        else:
            raise HTTPException(status_code=500, detail=result.get("message"))

    except asyncio.TimeoutError:
        raise HTTPException(
            status_code=504,
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
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="info")