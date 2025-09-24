import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import csv
import matplotlib.pyplot as plt
from serpapi import GoogleSearch

load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
except KeyError as e:
    print(f"FATAL ERROR: Environment variable {e} not set.")
    print("Please make sure you have a .env file with GEMINI_API_KEY and SERPAPI_API_KEY.")
    exit()

ai_judge_model = genai.GenerativeModel('gemini-1.5-pro-latest')

def google_search(query: str) -> str:
    """
    A tool that performs a Google search using the SERP API and returns the top results.
    """
    try:
        params = {
            "q": query,
            "api_key": SERPAPI_API_KEY
        }
        search = GoogleSearch(params)
        results = search.get_dict()

        snippets = []
        if "organic_results" in results:
            for result in results["organic_results"][:3]: # top 3 results
                snippet = result.get("snippet", "No snippet available.")
                snippets.append(f"Source: {result.get('link')}\nSnippet: {snippet}")
        
        if not snippets:
            return "No search results found."
            
        return "\n\n".join(snippets)

    except Exception as e:
        return f"Error during search: {e}"


def fact_checker_agent(prompt: str, response: str):
    """
    An agent that verifies the factual accuracy of a response, using a search tool.
    """
    print("└─ Running Fact-Checker Agent...")
    search_query = f"fact check: {response}"
    search_results = google_search(search_query)

    judge_prompt = f"""
    You are a meticulous Fact-Checker AI. Your task is to evaluate if the agent's answer is factually accurate based on the provided Google Search results.

    - First, analyze the Original Question and the Agent's Answer.
    - Second, use the provided Search Results as the single source of truth.
    - Finally, provide a score and a brief justification.

    Your score must be one of: "Correct", "Incorrect", or "Unverifiable".

    You must respond in a valid JSON format with two keys: "score" and "justification".
    Example: {{"score": "Incorrect", "justification": "The agent states the capital is Sydney, but search results confirm it is Canberra."}}

    Original Question: "{prompt}"
    Agent's Answer: "{response}"
    Search Results:
    ---
    {search_results}
    ---
    """
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
        judge_response = ai_judge_model.generate_content(
            judge_prompt,
            generation_config=generation_config
        )
        return json.loads(judge_response.text)
    except Exception as e:
        return {"score": "AI Judge Error", "justification": str(e)}

def reasoning_agent(prompt: str, response: str):
    """
    An agent that evaluates the logical reasoning of a response.
    """
    print("└─ Running Reasoning Agent...")
    judge_prompt = f"""
    You are a Logical Reasoning Evaluator. Your task is to analyze the logical steps and the process an agent took to arrive at its answer. Do not focus on the final answer's accuracy, but on the *method*.

    - Evaluate if the agent's reasoning is "Sound" (follows a logical path) or "Flawed" (contains logical errors, skips steps, or uses incorrect logic).
    - Provide a one-sentence justification explaining the logical step that was correct or incorrect.

    You must respond in a valid JSON format with two keys: "score" and "justification".
    Example: {{"score": "Flawed", "justification": "The agent ignored the step where 2 apples were eaten before adding the new ones."}}

    Original Question: "{prompt}"
    Agent's Answer: "{response}"
    """
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
        judge_response = ai_judge_model.generate_content(
            judge_prompt,
            generation_config=generation_config
        )
        return json.loads(judge_response.text)
    except Exception as e:
        return {"score": "AI Judge Error", "justification": str(e)}

def instruction_adherence_agent(prompt: str, response: str):
    """
    An agent that checks if specific instructions in the prompt were followed.
    """
    print("└─ Running Instruction-Adherence Agent...")

    if "single sentence" in prompt.lower():
        if response.strip().count('.') == 1 and response.strip().endswith('.'):
             return {"score": "Pass", "justification": "Rule-based check: The response is a single sentence."}
    
    judge_prompt = f"""
    You are an Instruction-Following an AI evaluator. Your task is to determine if the agent's answer followed all constraints and instructions given in the original question.

    - Identify any specific instructions (e.g., "in a single sentence", "in XML format", "as a summary").
    - If no instructions are present, the score is "Not Applicable".
    - Otherwise, score if the agent's response was a "Pass" or "Fail".
    - Provide a one-sentence justification.

    You must respond in a valid JSON format with two keys: "score" and "justification".
    Example: {{"score": "Fail", "justification": "The prompt asked for a single sentence, but the agent provided multiple."}}

    Original Question: "{prompt}"
    Agent's Answer: "{response}"
    """
    try:
        generation_config = genai.types.GenerationConfig(
            temperature=0.0,
            response_mime_type="application/json"
        )
        judge_response = ai_judge_model.generate_content(
            judge_prompt,
            generation_config=generation_config
        )
        return json.loads(judge_response.text)
    except Exception as e:
        return {"score": "AI Judge Error", "justification": str(e)}


def orchestrator(agent_name: str, prompt: str, response: str):
    """
    The orchestrator that runs all specialist agents and gathers their results.
    """
    print(f"--- Evaluating Response from '{agent_name}' ---")
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{response}'\n")

    results = {
        "Factual_Accuracy": fact_checker_agent(prompt, response),
        "Logical_Reasoning": reasoning_agent(prompt, response),
        "Instruction_Adherence": instruction_adherence_agent(prompt, response)
    }
    return results

def print_multi_agent_report(results: dict):
    """Prints the final, synthesized report from all agents."""
    print("Scores:")
    for category, result in results.items():
        score = result.get('score', 'N/A')
        justification = result.get('justification', 'N/A')
        print(f"- {category.replace('_', ' ')}: {score}")
        print(f"  └─ Justification: {justification}")
    print("-------------------------------------------\n")


def create_leaderboard_chart(agent_scores, filename="leaderboard.png"):
    """
    Creates and saves a bar chart visualizing the agent scores.
    """
    sorted_agents = sorted(agent_scores.items(), key=lambda item: item[1]['Factual_Correct'], reverse=True)
    agent_names = [agent[0] for agent in sorted_agents]
    

    factual_correct = [agent[1]['Factual_Correct'] for agent in sorted_agents]
    reasoning_sound = [agent[1]['Reasoning_Sound'] for agent in sorted_agents]
    instruction_pass = [agent[1]['Instruction_Pass'] for agent in sorted_agents]
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.25
    index = range(len(agent_names))


    ax.bar(index, factual_correct, bar_width, label='Factual Correct', color='skyblue')
    ax.bar([i + bar_width for i in index], reasoning_sound, bar_width, label='Reasoning Sound', color='lightgreen')
    ax.bar([i + 2 * bar_width for i in index], instruction_pass, bar_width, label='Instruction Pass', color='salmon')

    ax.set_xlabel('Agent Name', fontweight='bold')
    ax.set_ylabel('Number of Responses', fontweight='bold')
    ax.set_title('Multi-Agent Performance Leaderboard', fontweight='bold', fontsize=16)
    ax.set_xticks([i + bar_width for i in index])
    ax.set_xticklabels(agent_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nLeaderboard chart saved to {filename}")



def main():
    """
    Main function to run the multi-agent evaluation pipeline.
    """
    agent_scores = {}

    try:
        print("Starting Multi-Agent evaluation pipeline...")
        with open('data.csv', mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                agent = row["agent"]
                if agent not in agent_scores:
                    agent_scores[agent] = {
                        "Total": 0, "Errors": 0,
                        "Factual_Correct": 0, "Factual_Incorrect": 0,
                        "Reasoning_Sound": 0, "Reasoning_Flawed": 0,
                        "Instruction_Pass": 0, "Instruction_Fail": 0
                    }

                # The orchestrator runs the entire evaluation for one item
                results = orchestrator(row["agent"], row["prompt"], row["response"])
                
                # Print the synthesized report for the user
                print_multi_agent_report(results)

                # Aggregate scores for the final leaderboard
                agent_scores[agent]["Total"] += 1
                
                # Accuracy Score
                acc_score = results["Factual_Accuracy"].get('score')
                if acc_score == "Correct":
                    agent_scores[agent]["Factual_Correct"] += 1
                elif acc_score == "Incorrect":
                    agent_scores[agent]["Factual_Incorrect"] += 1
                
                # Reasoning Score
                reas_score = results["Logical_Reasoning"].get('score')
                if reas_score == "Sound":
                    agent_scores[agent]["Reasoning_Sound"] += 1
                elif reas_score == "Flawed":
                    agent_scores[agent]["Reasoning_Flawed"] += 1
                
                # Instruction Score
                inst_score = results["Instruction_Adherence"].get('score')
                if inst_score == "Pass":
                    agent_scores[agent]["Instruction_Pass"] += 1
                elif inst_score == "Fail":
                    agent_scores[agent]["Instruction_Fail"] += 1

        #  Final Leaderboard 
        print("\n===================================")
        print(" AI AGENTS LEADERBOARD (Multi-Agent)")
        print("===================================\n")

        sorted_agents = sorted(agent_scores.items(), key=lambda item: item[1]['Factual_Correct'], reverse=True)

        for agent, scores in sorted_agents:
            print(f"--- Agent: {agent} ({scores['Total']} responses) ---")
            print(f"  - Factual Accuracy: {scores['Factual_Correct']} Correct, {scores['Factual_Incorrect']} Incorrect")
            print(f"  - Logical Reasoning: {scores['Reasoning_Sound']} Sound, {scores['Reasoning_Flawed']} Flawed")
            print(f"  - Instruction Following: {scores['Instruction_Pass']} Pass, {scores['Instruction_Fail']} Fail")
            print()
            
        create_leaderboard_chart(agent_scores)

    except FileNotFoundError:
        print("FATAL ERROR: data.csv not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()