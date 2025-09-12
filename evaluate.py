import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import csv

load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
except KeyError:
    print("Error: GEMINI_API_KEY environment variable not set.")
    print("Please make sure you have a .env file with your API key.")
    exit()

ai_judge_model = genai.GenerativeModel('gemini-1.5-pro-latest')



def evaluate_instruction_following(prompt, response):
    """
    A precise, rule-based check for a "single sentence" instruction.
    """
    if "single sentence" in prompt.lower():
        if response.strip().count('.') == 1 and response.strip().endswith('.'):
            return "PASS"
        else:
            return "FAIL"
    return "Not Applicable"

def evaluate_helpfulness_with_ai(prompt, response):
    """
    Evaluates helpfulness and provides a justification for the score using the AI Judge.
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
            response_mime_type="application/json"  # Force the model to output JSON
        )
        judge_response = ai_judge_model.generate_content(
            judge_prompt,
            generation_config=generation_config
        )
        return json.loads(judge_response.text)
    except Exception as e:
        return {"score": "AI Judge Error", "justification": str(e)}

def run_and_print_evaluation(agent_name, prompt, response):
    """
    Runs all evaluations, prints a detailed report for one item,
    and returns the scores for aggregation.
    """
    instruction_score = evaluate_instruction_following(prompt, response)
    helpfulness_eval = evaluate_helpfulness_with_ai(prompt, response)

    print("--- Evaluation Report ---")
    print(f"Agent: '{agent_name}'")
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{response}'\n")
    print("Scores:")
    print(f"- Instruction Following: {instruction_score}")
    print(f"- Helpfulness: {helpfulness_eval.get('score', 'N/A')}")
    print(f"  └─ Justification: {helpfulness_eval.get('justification', 'N/A')}")
    print("-------------------------\n")

    return instruction_score, helpfulness_eval



def main():
    """
    Main function to run the evaluation pipeline.
    It reads data from a CSV, evaluates each item, and prints a final leaderboard.
    """
    agent_scores = {}

    try:
        print("Starting evaluation pipeline...")
        with open('data.csv', mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                agent = row["agent"]
                if agent not in agent_scores:
                    agent_scores[agent] = {"Helpful": 0, "Unhelpful": 0, "Fail": 0, "Total": 0, "Errors": 0}

            
                instruction_score, helpfulness_eval = run_and_print_evaluation(
                    row["agent"], row["prompt"], row["response"]
                )

                # leaderboard scores
                agent_scores[agent]["Total"] += 1
                if instruction_score == "FAIL":
                    agent_scores[agent]["Fail"] += 1
                
                score = helpfulness_eval.get('score')
                if score == "Helpful":
                    agent_scores[agent]["Helpful"] += 1
                elif score == "Unhelpful":
                    agent_scores[agent]["Unhelpful"] += 1
                elif score == "AI Judge Error":
                    agent_scores[agent]["Errors"] += 1


        print("\n===================================")
        print(" AI AGENTS LEADERBOARD")
        print("===================================\n")

        sorted_agents = sorted(agent_scores.items(), key=lambda item: item[1]['Helpful'], reverse=True)

        for agent, scores in sorted_agents:
            print(f"--- Agent: {agent} ({scores['Total']} responses) ---")
            print(f"  - Helpful Responses: {scores['Helpful']}")
            print(f"  - Unhelpful Responses: {scores['Unhelpful']}")
            print(f"  - Instruction Fails: {scores['Fail']}")
            if scores['Errors'] > 0:
                print(f"  - AI Judge Errors: {scores['Errors']}")
            print()

    except FileNotFoundError:
        print("FATAL ERROR: data.csv not found. Please create it in the same folder as the script.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()