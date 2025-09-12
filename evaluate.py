import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set it to your Google AI API key.")
    exit()

# AI model 
ai_judge_model = genai.GenerativeModel('gemini-1.5-pro-latest')


def evaluate_instruction_following(prompt, response):
    """
    A more precise rule-based check for instruction following.
    """
    if "single sentence" in prompt.lower():
        # A better heuristic: a single sentence has one period and ends with it.
        if response.strip().count('.') == 1 and response.strip().endswith('.'):
            return "PASS"
        else:
            return "FAIL" 
    return "Not Applicable"

def evaluate_helpfulness_with_ai(prompt, response):
    """
    Evaluates helpfulness and provides a justification for the score.
    This targets the "Explainability" stretch goal.
    """
    judge_prompt = f"""
    You are an AI quality evaluator. Your task is to assess if the agent's answer is helpful and relevant to the user's question, and provide a brief justification for your score.

    First, evaluate the agent's answer based on these criteria:
    - **Helpful**: The answer directly and usefully addresses the user's question.
    - **Unhelpful**: The answer attempts to address the question but is factually wrong, incomplete, or of very poor quality.
    - **Irrelevant**: The answer does not address the user's question at all.

    Second, provide a one-sentence justification for your evaluation.

    You must respond in a valid JSON format with two keys: "score" and "justification".
    Example: {{"score": "Helpful", "justification": "The answer correctly identifies the boiling point of water."}}

    Original Question: "{prompt}"
    Agent's Answer: "{response}"
    """
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json" # This forces the model to output JSON!
        )
        judge_response = ai_judge_model.generate_content(
            judge_prompt,
            generation_config=generation_config
        )
        return json.loads(judge_response.text)
    except Exception as e:
        # If something goes wrong, return a structured error
        return {"score": "AI Judge Error", "justification": str(e)}

def run_evaluation(agent_name, prompt, response):
    """ Runs all evaluations and prints the score with justification. """
    print("--- Evaluation Report ---")
    print(f"Agent: '{agent_name}'")
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{response}'\n")

    instruction_score = evaluate_instruction_following(prompt, response)
    helpfulness_eval = evaluate_helpfulness_with_ai(prompt, response) 

    print("Scores:")
    print(f"- Instruction Following: {instruction_score}")
    print(f"- Helpfulness: {helpfulness_eval['score']}")
    print(f"  └─ Justification: {helpfulness_eval['justification']}") 
    print("-------------------------\n")

#  Example evaluation data

eval_data = [
    {
        "agent": "HelpfulBot",
        "prompt": "What is the boiling point of water in Celsius?",
        "response": "The boiling point of water is 100 degrees Celsius."
    },
    {
        "agent": "CreativeBot",
        "prompt": "Explain gravity in a single sentence.",
        "response": "Gravity is the invisible force that keeps everything from floating away. It's what makes apples fall from trees and keeps us on the ground. It is a fundamental interaction of nature."
    },
    {
        "agent": "WrongBot",
        "prompt": "What is the capital of Australia?",
        "response": "The capital of Australia is Sydney."
    }
]


for item in eval_data:
    run_evaluation(item["agent"], item["prompt"], item["response"])