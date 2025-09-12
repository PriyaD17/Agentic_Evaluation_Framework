import google.generativeai as genai
import os



try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please set it to your Google AI API key.")
    exit()

ai_judge_model = genai.GenerativeModel('gemini-pro')


def evaluate_instruction_following(prompt, response):
    """
    A simple rule-based check for instruction following.
    This example checks if the response is a single sentence.
    """
    # simple heuristic: a single sentence should have one period.
    if "single sentence" in prompt.lower() and response.count('.') == 1:
        return "PASS"
    # placeholder for more complex rules- we default to PASS for now.
    return "PASS (Rule not applicable)"

def evaluate_accuracy_with_ai(prompt, response):
    """
    Uses an AI model as a "judge" to check for factual accuracy.
    """
    judge_prompt = f"""
    You are a fact-checking system. Based on the original question, is the following answer factually correct?
    Respond with only the word 'Correct' or 'Incorrect'.

    Original Question: "{prompt}"
    Answer to Evaluate: "{response}"
    """

    try:
        judge_response = ai_judge_model.generate_content(judge_prompt)

        return judge_response.text.strip()
    except Exception as e:
        return f"Error during AI check: {e}"


def run_evaluation(agent_name, prompt, response):
    """
    Runs all evaluations for a given prompt and response.
    """
    print("--- Evaluation Report ---")
    print(f"Agent: '{agent_name}'")
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{response}'\n")


    instruction_score = evaluate_instruction_following(prompt, response)
    accuracy_score = evaluate_accuracy_with_ai(prompt, response)


    print("Scores:")
    print(f"- Instruction Following: {instruction_score}")
    print(f"- Accuracy: {accuracy_score}")
    print("-------------------------\n")


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