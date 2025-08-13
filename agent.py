# agent.py

import os
import traceback
import io
import base64
from typing import List, Dict, Any

# FIX: Set matplotlib backend to a non-GUI one to avoid environment errors.
# This must be done BEFORE importing pyplot.
import matplotlib
matplotlib.use('Agg')

# Gemini specific library
import google.generativeai as genai

# Tool Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import duckdb
import re # Import re for cleaning and sanitizing

# Configure the Gemini client with the API key from environment variables
genai.configure(api_key=os.environ["GEMINI_API_KEY"])

class DataAnalystAgent:
    """The core agent that generates and executes code to answer questions using Gemini."""

    def __init__(self, max_retries: int = 3):
        self.max_retries = max_retries
        # Initialize the Gemini model with the master system prompt
        self.model = genai.GenerativeModel(
            model_name="gemini-2.5-pro",
            system_instruction=self._get_system_prompt()
        )
        self.last_debug_prompt = ""

    def _convert_to_native_types(self, data):
        if isinstance(data, list):
            return [self._convert_to_native_types(item) for item in data]
        if isinstance(data, dict):
            return {k: self._convert_to_native_types(v) for k, v in data.items()}
        if isinstance(data, np.integer):
            return int(data)
        if isinstance(data, np.floating):
            return float(data)
        if isinstance(data, np.bool_):
            return bool(data)
        return data

    def run(self, question: str, uploaded_files: List[Any]) -> Dict[str, Any]:
        """
        The main execution loop with self-healing capabilities.
        """
        file_context = self._prepare_file_context(uploaded_files)

        user_prompt = f"""
        User's question:
        ---
        {question}
        ---

        You have been provided with the following data files, which are loaded into variables: {list(file_context.keys())}.
        Please generate a Python script to answer the user's question.
        """

        chat_session = self.model.start_chat(history=[])

        for attempt in range(self.max_retries):
            # FIX: Initialize generated_code to prevent UnboundLocalError if the API call itself fails.
            generated_code = ""
            print(f"--- Agent Attempt #{attempt + 1} ---")
            prompt_to_send = user_prompt if attempt == 0 else self.last_debug_prompt

            try:
                response = chat_session.send_message(prompt_to_send)
                generated_code = response.text

                if "```python" in generated_code:
                    generated_code = generated_code.split("```python\n")[1].split("\n```")[0]

                print("Generated Code:\n", generated_code)

                execution_scope = self._create_execution_scope(file_context)
                exec(generated_code, execution_scope)

                result = execution_scope.get("result")
                print("Execution Successful. Raw Result:", result)
                
                serializable_result = self._convert_to_native_types(result)
                print("Execution Successful. Serializable Result:", serializable_result)

                return {"status": "success", "result": serializable_result}

            except Exception as e:
                print(f"Execution failed on attempt {attempt + 1}: {e}")
                error_traceback = traceback.format_exc()

                if attempt == self.max_retries - 1:
                    print("Max retries reached. Returning final error.")
                    return {"status": "error", "message": f"Agent failed after {self.max_retries} attempts. Last error: {error_traceback}"}

                self.last_debug_prompt = f"""
                The previous attempt failed.

                Faulty Code:
                ---
                {generated_code}
                ---

                Error Message:
                ---
                {error_traceback}
                ---

                Please analyze the error and provide a corrected Python script.
                """

        return {"status": "error", "message": "Agent failed to produce a working script."}

    def _prepare_file_context(self, uploaded_files: List[Any]) -> Dict[str, Any]:
        """Reads uploaded files and prepares them as pandas DataFrames or BytesIO objects."""
        context = {}
        if not uploaded_files:
            return context

        for file in uploaded_files:
            try:
                variable_name = re.sub(r'[^a-zA-Z0-9_]', '_', file.filename)
                if file.filename.endswith('.csv'):
                    content = file.file.read()
                    context[variable_name] = pd.read_csv(io.BytesIO(content))
                else:
                    content = file.file.read()
                    context[variable_name] = io.BytesIO(content)
                file.file.seek(0)
            except Exception as e:
                print(f"Error reading file {file.filename}: {e}")
        return context

    def _create_execution_scope(self, file_context: Dict[str, Any]) -> Dict[str, Any]:
        """Creates the global scope for the exec() function."""
        scope = {
            "pd": pd, "np": np, "plt": plt, "sns": sns,
            "duckdb": duckdb, "requests": __import__('requests'),
            "BeautifulSoup": __import__('bs4').BeautifulSoup,
            "io": io, "base64": base64, "re": re,
            "result": None,
        }
        scope.update(file_context)
        return scope

    def _get_system_prompt(self) -> str:
        """The master system prompt defining the agent's behavior for Gemini."""
        return """
        **Role:** You are an AI assistant that specializes in breaking down complex questions into programmable Python code.

        **Task:** Given a question, generate a Python script to perform the requested data analysis.

        **Column Name Mandate:**
        - You **MUST NOT** assume or hardcode column names (e.g., 'Sales', 'Region', 'Date'). The actual names in the uploaded file may vary (e.g., ' Sales Amount ', 'location', 'order_date').
        - Your code **MUST** programmatically identify the correct column names. First, clean all column names in the DataFrame, then search for keywords to find the columns you need.
        - **EXAMPLE CODE TEMPLATE for finding columns:**
          ```python
          # Clean column names (strip whitespace, etc.)
          df.columns = df.columns.str.strip()
          
          # Dynamically find column names based on keywords
          sales_col = next((col for col in df.columns if 'sale' in col.lower()), None)
          region_col = next((col for col in df.columns if 'region' in col.lower() or 'location' in col.lower()), None)
          date_col = next((col for col in df.columns if 'date' in col.lower()), None)
          
          # Raise an error if essential columns are not found
          if not sales_col:
              raise ValueError(f"Could not find a 'sales' column. Actual columns: {df.columns.tolist()}")
          ```

        **Execution Environment:**
        - Libraries available: `pandas as pd`, `numpy as np`, `matplotlib.pyplot as plt`, `seaborn as sns`, `duckdb`, `requests`, `bs4.BeautifulSoup`, `io`, `base64`, and `re`.
        - Uploaded files are available as variables named after their sanitized filenames (e.g., a file 'my-data.csv' is available as a DataFrame named 'my_data_csv').

        **Data Scraping Rules:**

        **Data Cleaning Rules:**
        - Before converting a column to a numeric type, you **MUST** clean it by removing non-numeric characters (e.g., $, ,, M, K, RK). Use `regex=True`.
        
        **Output Formatting Rules:**
        - The final answer **MUST** be stored in a variable named `result`.
        - The `result` variable **MUST** be a Python list containing only raw data values (integers, floats, strings), not descriptive sentences.

        **Plotting Rules:**
        - Generate plots using matplotlib, save to an in-memory buffer, encode as a base64 string, and format as a data URI (`data:image/png;base64,...`) under 100,000 bytes.

        **Debugging Rules:**
        - If your code fails, analyze the error and provide a fully corrected script.
        
        Respond ONLY with the Python code, enclosed in ```python ... ```.
        """