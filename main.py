from fastapi import FastAPI, HTTPException, Query
from pathlib import Path
import requests
import json
import os
import functions

app = FastAPI()

DATA_DIR = Path("/data")
AIPROXY_URL = "https://aiproxy.sanand.workers.dev/openai/v1/chat/completions"
AIPROXY_TOKEN = os.getenv("AIPROXY_TOKEN")

# Mapping of known tasks to functions
TASK_FUNCTIONS = {
    "run_datagen": functions.run_datagen,
    "format_markdown": functions.format_markdown,
    "count_wednesdays": functions.count_wednesdays,
    "extract_recent_logs":functions.extract_recent_logs,
    "index_markdown_titles": functions.index_markdown_titles,
    "extract_email_sender": functions.extract_email_sender,
    "process_credit_card_image": functions.process_credit_card_image,
    "find_most_similar_comments": functions.find_most_similar_comments,
    "calculate_gold_ticket_sales": functions.calculate_gold_ticket_sales,
}


def query_llm(task_description: str) -> str:
    """Uses AI Proxy to interpret the task and return the internal function name."""
    headers = {
        "Authorization": f"Bearer {AIPROXY_TOKEN}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gpt-4o-mini",
        "messages": [
            {"role": "system", "content": "You are an automation agent. Map the given task description to one of these functions: " + ", ".join(TASK_FUNCTIONS.keys()) + "."},
            {"role": "user", "content": task_description}
        ],
        "temperature": 0
    }

    try:
        response = requests.post(AIPROXY_URL, json=data, headers=headers)
        response.raise_for_status()
        function_name = response.json()["choices"][0]["message"]["content"].strip()
        return function_name if function_name in TASK_FUNCTIONS else None
    except Exception as e:
        print(f"❌ AI Proxy Error: {e}")
        return None


@app.post("/run")
async def run_task(task: str = Query(..., description="Task description in plain English")):
    """Executes a task based on the provided description."""
    function_name = query_llm(task)

    if not function_name:
        raise HTTPException(status_code=400, detail="❌ Could not understand the task.")

    try:
        result = TASK_FUNCTIONS[function_name]()  # Call the function from functions.py
        return {"status": "success", "output": result}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Internal error: {str(e)}")


@app.get("/read")
async def read_file(path: str = Query(..., description="File path to read")):
    """Returns the content of a given file."""
    file_path = DATA_DIR / path.lstrip("/")

    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    try:
        return file_path.read_text(encoding="utf-8")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"❌ Error reading file: {e}")
    


