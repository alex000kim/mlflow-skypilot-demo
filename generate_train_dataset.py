import json
from datasets import load_dataset

# Define the system message
system_message = """
Solve the given high school math problem by providing a clear and detailed explanation.
Provide a detailed breakdown of your calculations, beginning with an explanation of the problem.
# Steps
1. **Understand the Problem**: Restate the given math problem and clearly identify its requirements.
2. **Set Up**: Identify the key formulas or concepts that could help solve the problem.
3. **Solve Step-by-Step**: Iteratively progress through each step of the math problem, providing explanations.
4. **Double Check**: If applicable, double-check the work for accuracy and sense.
5. **Final Answer**: Provide the numerical or algebraic solution clearly, accompanied by the final reasoning.
# Notes
- Always clearly define any variable or term used.
- Wherever applicable, include unit conversions or context to explain why each formula or step is applied.
- Assume the level of mathematics is suitable for high school, and avoid overly complex methods.
"""

# Function to create conversations
def create_conversation(sample):
    return {
        "messages": [
            {"role": "system", "content": system_message},
            {"role": "user", "content": sample["question"]},
            {"role": "assistant", "content": sample["answer"]},
        ]
    }

def main():
    # Load Orca-Math dataset
    dataset = load_dataset("microsoft/orca-math-word-problems-200k", split="train")

    # Convert dataset to the required format
    formatted_dataset = dataset.map(create_conversation, remove_columns=dataset.column_names)

    # Save the formatted dataset to JSON
    formatted_data = [sample for sample in formatted_dataset]
    output_path = "train_dataset.json"

    with open(output_path, "w") as json_file:
        json.dump(formatted_data, json_file, indent=2)

    print(f"Dataset saved to {output_path}")

if __name__ == "__main__":
    main()
