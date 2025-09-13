# Agentic Evaluation Framework - e6data Hackathon

This project is a complete, end-to-end framework for evaluating the performance of AI agents across multiple, nuanced dimensions. It goes beyond simple "accuracy" to measure "helpfulness" and provides AI-powered justifications for its scores.

## The Problem
Evaluating AI agents is challenging. A simple "right or wrong" score is not enough because an agent can be factually correct but fail to follow instructions, or it can be creative and helpful without being a textbook definition. We need a system that can capture this nuance automatically and at scale.

## The Solution
Built a multi-dimensional evaluation pipeline that scores agents on two key axes:
1.  **Rule-Based Instruction Following:** A precise, code-based check to see if the agent followed specific constraints (e.g., "respond in a single sentence").
2.  **AI-Based Quality Scoring:** Using Gemini 1.5 Pro as an "AI Judge," the framework evaluates the **helpfulness** and **relevance** of the response from a user's perspective.

### Key Features
*   **Explainability:** The AI Judge provides a one-sentence justification for every score, explaining *why* a response was deemed helpful or unhelpful.
*   **Scalability:** The pipeline reads data from an external `.csv` file, allowing it to evaluate thousands of responses.
*   **Multi-Domain Support:** The framework was successfully tested on Question-Answering, Summarization, and Reasoning tasks.
*   **Automated Leaderboard & Visualization:** The script generates a final text-based leaderboard and a visual bar chart (`leaderboard.png`) for an at-a-glance understanding of agent performance.

## How to Run It
1.  **Prerequisites:** Python 3.7+
2.  **Install dependencies:**
    ```bash
    pip install google-generativeai python-dotenv matplotlib
    ```
3.  **Set up your API Key:**
    *   Create a `.env` file in the root directory.
    *   Add your Google AI API key: `GEMINI_API_KEY="YOUR_KEY_HERE"`
4.  **Run the pipeline:**
    ```bash
    python evaluate.py
    ```

## Accomplishments & Issues Faced
During this hackathon, the entire pipeline was built from scratch. A key challenge was evolving our evaluation metric from simple "accuracy" to "helpfulness" after discovering that powerful AI judges are very literal. By refining our prompts and our logic, we created a more robust and insightful system.