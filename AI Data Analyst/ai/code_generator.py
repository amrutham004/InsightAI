
import ollama


MODEL = "llama3.2"


def generate_code(
    domain,
    task,
    question,
    dataset_columns=None
):

    columns = dataset_columns or []

    system_prompt = f"""
You are InsightAI, a local AI coding assistant.

Your programming domain is: {domain}
Your requested task is: {task}

Generate accurate, beginner-friendly code.

Requirements:

1. Generate complete code when appropriate.
2. Explain the code in simple English.
3. Include required libraries.
4. Include assumptions.
5. Use the user's actual column names when provided.
6. Never invent dataset columns.
7. Include error handling where appropriate.
8. Avoid unnecessary dependencies.
9. Never execute generated code.
10. Do not request external AI APIs.

Return your response in this format:

## Understanding
Explain the task.

## Required Libraries
List the libraries.

## Code
Provide the code in a code block.

## Explanation
Explain the code.

## How to Use
Explain how to run or apply it.
"""

    user_prompt = f"""
User question:
{question}

Additional task details:
{task}

Dataset columns:
{columns}
"""

    response = ollama.chat(
        model=MODEL,
        messages=[
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": user_prompt
            }
        ]
    )

    return response.message.content
