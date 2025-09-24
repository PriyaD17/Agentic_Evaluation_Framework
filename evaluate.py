import google.generativeai as genai
import os
from dotenv import load_dotenv
import json
import csv
import matplotlib.pyplot as plt
from serpapi import GoogleSearch
import time

load_dotenv()

try:
    genai.configure(api_key=os.environ["GEMINI_API_KEY"])
    SERPAPI_API_KEY = os.environ["SERPAPI_API_KEY"]
except KeyError as e:
    print(f"FATAL ERROR: Environment variable {e} not set.")
    print("Please make sure you have a .env file with GEMINI_API_KEY and SERPAPI_API_KEY.")
    exit()

ai_judge_model = genai.GenerativeModel('gemini-1.5-pro-latest')
def estimate_token_count(text: str) -> int:
    """
    A simple heuristic to estimate token count. A real implementation might use
    a proper tokenizer library or get the count from the API response.
    A common ratio is ~4 characters per token.
    """
    return round(len(text) / 4)

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

def conciseness_agent(prompt: str, response: str):
    """
    **NEW AGENT**: An agent that evaluates the conciseness of a response.
    """
    print("└─ Running Conciseness Agent...")
    judge_prompt = f"""
    You are a Conciseness Evaluator. Your task is to determine if the agent's answer is brief and to the point, or if it is overly verbose.

    - Evaluate the response as "Concise" if it answers the question without unnecessary words or information.
    - Evaluate the response as "Verbose" if it contains filler or irrelevant details that make the answer longer than necessary.

    You must respond in a valid JSON format with two keys: "score" and "justification".
    Example: {{"score": "Verbose", "justification": "The agent gave a long history of water before stating its boiling point."}}

    Original Question: "{prompt}"
    Agent's Answer: "{response}"
    """
    try:
        generation_config = genai.types.GenerationConfig(temperature=0.0, response_mime_type="application/json")
        judge_response = ai_judge_model.generate_content(judge_prompt, generation_config=generation_config)
        return json.loads(judge_response.text)
    except Exception as e:
        return {"score": "AI Judge Error", "justification": str(e)}

def orchestrator(agent_name: str, prompt: str, response: str) -> dict:
    """
    The orchestrator now implements a Communication Protocol by creating and
    populating a structured "Task State" object for each evaluation.
    """
    print(f"--- Evaluating Response from '{agent_name}' ---")
    print(f"Prompt: '{prompt}'")
    print(f"Response: '{response}'\n")
    
    start_time = time.time()

    task_state = {
        "inputs": {
            "agent_name": agent_name,
            "prompt": prompt,
            "response": response
        },
        "metadata": {
            "latency_seconds": 0,
            "estimated_cost_tokens": 0
        },
        "results": {}
    }

    task_state["results"]["Factual_Accuracy"] = fact_checker_agent(prompt, response)
    task_state["results"]["Logical_Reasoning"] = reasoning_agent(prompt, response)
    task_state["results"]["Instruction_Adherence"] = instruction_adherence_agent(prompt, response)
    task_state["results"]["Conciseness"] = conciseness_agent(prompt, response) # New agent call

    end_time = time.time()
    task_state["metadata"]["latency_seconds"] = round(end_time - start_time, 2)
    
    # Estimate total tokens used for all agent prompts and the original response
    total_text_for_cost = prompt + response + json.dumps(task_state["results"])
    task_state["metadata"]["estimated_cost_tokens"] = estimate_token_count(total_text_for_cost)

    return task_state

def print_multi_agent_report(task_state: dict):
    """Prints the final, synthesized report from the Task State."""
    print("Scores:")
    for category, result in task_state["results"].items():
        score = result.get('score', 'N/A')
        justification = result.get('justification', 'N/A')
        print(f"- {category.replace('_', ' ')}: {score}")
        print(f"  └─ Justification: {justification}")
    
    print("\nPerformance Metrics:")
    print(f"- Latency: {task_state['metadata']['latency_seconds']} seconds")
    print(f"- Estimated Cost: {task_state['metadata']['estimated_cost_tokens']} tokens")
    print("-------------------------------------------\n")


def create_leaderboard_chart(agent_scores, filename="leaderboard.png"):
    """Creates and saves a bar chart visualizing the agent scores."""
    sorted_agents = sorted(agent_scores.items(), key=lambda item: item[1]['Factual_Correct'], reverse=True)
    agent_names = [agent[0] for agent in sorted_agents]
    
    factual_correct = [agent[1]['Factual_Correct'] for agent in sorted_agents]
    reasoning_sound = [agent[1]['Reasoning_Sound'] for agent in sorted_agents]
    instruction_pass = [agent[1]['Instruction_Pass'] for agent in sorted_agents]
    concise_scores = [agent[1]['Concise'] for agent in sorted_agents] 

    fig, ax = plt.subplots(figsize=(12, 7))
    bar_width = 0.2
    index = range(len(agent_names))

    ax.bar(index, factual_correct, bar_width, label='Factual Correct', color='skyblue')
    ax.bar([i + bar_width for i in index], reasoning_sound, bar_width, label='Reasoning Sound', color='lightgreen')
    ax.bar([i + 2 * bar_width for i in index], instruction_pass, bar_width, label='Instruction Pass', color='salmon')
    ax.bar([i + 3 * bar_width for i in index], concise_scores, bar_width, label='Concise', color='purple') 

    ax.set_xlabel('Agent Name', fontweight='bold')
    ax.set_ylabel('Number of Responses', fontweight='bold')
    ax.set_title('Multi-Agent Performance Leaderboard', fontweight='bold', fontsize=16)
    ax.set_xticks([i + 1.5 * bar_width for i in index])
    ax.set_xticklabels(agent_names, rotation=15, ha="right")
    ax.legend()
    ax.grid(axis='y', linestyle='--', alpha=0.7)

    plt.tight_layout()
    plt.savefig(filename)
    print(f"\nLeaderboard chart saved to {filename}")



def main():
    """Main function to run the multi-agent evaluation pipeline."""
    agent_scores = {}
    performance_metrics = {"total_latency": 0, "total_cost": 0}

    try:
        print("Starting Multi-Agent evaluation pipeline with enhanced metrics...")
        with open('data.csv', mode='r', encoding='utf-8') as infile:
            reader = csv.DictReader(infile)
            for row in reader:
                agent = row["agent"]
                if agent not in agent_scores:
                    agent_scores[agent] = {
                        "Total": 0, "Errors": 0,
                        "Factual_Correct": 0, "Factual_Incorrect": 0,
                        "Reasoning_Sound": 0, "Reasoning_Flawed": 0,
                        "Instruction_Pass": 0, "Instruction_Fail": 0,
                        "Concise": 0, "Verbose": 0 # New tracker
                    }

                task_state = orchestrator(row["agent"], row["prompt"], row["response"])
                print_multi_agent_report(task_state)

                agent_scores[agent]["Total"] += 1
                performance_metrics["total_latency"] += task_state["metadata"]["latency_seconds"]
                performance_metrics["total_cost"] += task_state["metadata"]["estimated_cost_tokens"]
                
                results = task_state["results"]
                
                # Accuracy Score
                if results["Factual_Accuracy"].get('score') == "Correct": agent_scores[agent]["Factual_Correct"] += 1
                # Reasoning Score
                if results["Logical_Reasoning"].get('score') == "Sound": agent_scores[agent]["Reasoning_Sound"] += 1
                # Instruction Score
                if results["Instruction_Adherence"].get('score') == "Pass": agent_scores[agent]["Instruction_Pass"] += 1
                # Conciseness Score
                if results["Conciseness"].get('score') == "Concise": agent_scores[agent]["Concise"] += 1

        # --- Final Leaderboard ---
        print("\n===================================")
        print(" AI AGENTS LEADERBOARD")
        print("===================================\n")

        sorted_agents = sorted(agent_scores.items(), key=lambda item: item[1]['Factual_Correct'], reverse=True)

        for agent, scores in sorted_agents:
            print(f"--- Agent: {agent} ({scores['Total']} responses) ---")
            print(f"  - Factual Correct: {scores['Factual_Correct']}")
            print(f"  - Reasoning Sound: {scores['Reasoning_Sound']}")
            print(f"  - Instruction Pass: {scores['Instruction_Pass']}")
            print(f"  - Concise: {scores['Concise']}")
            print()
            
        create_leaderboard_chart(agent_scores)

        print("\n--- Overall Performance Summary ---")
        print(f"Total evaluation time: {performance_metrics['total_latency']:.2f} seconds")
        print(f"Total estimated cost: {performance_metrics['total_cost']} tokens")
        print("-----------------------------------")


    except FileNotFoundError:
        print("FATAL ERROR: data.csv not found.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


if __name__ == "__main__":
    main()