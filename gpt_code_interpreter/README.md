# GPT Code Interpreter

This project contains scripts to create and execute agents that can interpret and run Python code, as well as answer questions about a CSV file.

## Scripts

- **main.py**
  - Contains the main function which sets up and runs different agents:
    - A Python agent that can write and execute Python code to answer questions.
    - A CSV agent that can answer questions about a CSV file using Python pandas.
    - A grand agent that routes questions to the appropriate specialized agent based on the input.