import pandas as pd
import random
import openai
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

# File paths
train_data_path = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\train_data\train_data.xlsx"
input_file = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\test_data\test_data.xlsx"
output_file = r"C:\Users\krupa\OneDrive\Documents\PSU\spring_24\Research\Data_Scoring\new_project\results\results_gpt_3.5_o1.csv"

# Load the training data
train_data = pd.read_excel(train_data_path)

# Function to generate the prompt with dynamic examples
def generate_prompt_with_examples(problem_id):
    # Subset the training data for the given problem_id
    subset = train_data[train_data["problem_id"] == problem_id]
    problem_description = subset["problem"].iloc[0]  # All descriptions for the same ID are assumed to be the same

    # Create the examples dynamically from the subset
    examples_text = "\n".join(
        [
            f"{i+1}. Response: {row.response} \nOriginality Score: {row.originality_score} \nExplanation: {row.explanation}"
            for i, row in enumerate(subset.itertuples(), start=1)
        ]
    )

    # Create the full system instructions prompt
    prompt_text = f"""
    System Instructions: 
    You will be given a problem and a response. The problem describes a challenge, and the response proposes a solution. A score will be assigned to the originality of the response, along with an explanation for that score.
    Your task is to provide an originality score to the response with respect to the originality of the response. You should also provide an explanation as to why you assigned that score to the response.

    When assessing originality, consider three key aspects:
    1. Novelty: How unique is the approach compared to typical solutions?
    2. Imagination: How creative and inventive is the idea?
    3. Structure: Does the idea transcend the given scenario, questioning assumptions and demonstrating out-of-the-box thinking?

    Originality Score Scale:
    1 - Very Unoriginal (very simple and/or common idea)
    2 - Unoriginal (simple idea, not novel or imaginative, structured by the scenario)
    3 - Neutral (limited novelty or imagination, still structured by the scenario)
    4 - Original (shows novelty and imagination, less structured by the problem)
    5 - Very Original (novel and imaginative, not structured by the problem)

    Evaluation Guidelines:
    - Rate the solution holistically, even if it contains multiple parts or ideas
    - Do not 'read into' the idea or assume unstated intentions
    - Rate based solely on what is written
    - Remember that originality is independent from quality

    Examples:
    Problem: {problem_description}
    {examples_text}
    """
    return prompt_text

# Function to send the prompt to OpenAI API and include the user message
def send_prompt(problem, response):
    # Generate a random problem_id and create the corresponding prompt
    random_problem_id = random.choice(train_data["problem_id"].unique())
    system_prompt = generate_prompt_with_examples(random_problem_id)

    # Create the user message
    user_message = f"Problem: {problem}\nResponse: {response}"

    # Send to OpenAI API
    response = openai.ChatCompletion.create(
        model="gpt-3.5-o1",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_message},
        ],
        max_tokens=1500,
    )

    # Extract the content of the API response
    api_response = response.choices[0].message["content"]
    return api_response.strip(), api_response

# Function to parse the output for originality score and explanation
def parse_output(problem, response, output):
    originality_score = "N/A"
    explanation = "N/A"

    # Attempt to parse originality score and explanation
    if "Originality Score:" in output:
        score_start = output.find("Originality Score:") + len("Originality Score:")
        score_end = output.find('\n', score_start)
        originality_score = output[score_start:score_end].strip()

    if "Explanation:" in output:
        explanation_start = output.find("Explanation:") + len("Explanation:")
        explanation_end = output.find('\n', explanation_start)
        explanation = output[explanation_start:explanation_end].strip()

    return {
        "problem": problem,
        "response": response,
        "originality_score": originality_score,
        "explanation": explanation
    }

# Function to process the test data and save results
def process_test_data():
    # Load the test data
    data = pd.read_excel(input_file)
    results = []

    # Iterate through each problem-response pair in the test data
    for index, row in data.iterrows():
        problem = row["problem"]
        response = row["response"]
        output, full_response = send_prompt(problem, response)
        result = parse_output(problem, response, output)
        results.append(result)

    # Save results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv(output_file, index=False)
    print("Data processed and saved successfully.")

# Main execution
if __name__ == "__main__":
    process_test_data()
