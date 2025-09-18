# Braintrust Random Data Generator

A Python script that generates random experiments and datasets using the Braintrust API for testing and development purposes.

## Features

- **Random Project Creation**: Creates projects with realistic names combining tasks and domains
- **Experiment Generation**: Generates experiments with random data, scores, and metadata
- **Dataset Creation**: Creates datasets with training/evaluation data
- **Realistic Data**: Uses sample tasks like text classification, question answering, summarization, etc.
- **Multiple Metrics**: Generates random scores for accuracy, precision, recall, F1, BLEU, ROUGE, etc.
- **Rate Limiting**: Built-in delays to respect API limits

## Setup

1. Install dependencies:
```bash
pip install requests
```

2. Set your Braintrust API key:
```bash
export BRAINTRUST_API_KEY="your_api_key_here"
```

3. Get your API key from the Braintrust organization settings page at https://www.braintrust.dev/

## Usage

### Basic Usage
Run the script to generate a complete setup:
```bash
python3 braintrust_generator.py
```

This creates:
- 1 random project
- 2 experiments with 15 records each
- 2 datasets with 25 records each

### Programmatic Usage

```python
from braintrust_generator import BraintrustGenerator

# Initialize generator
generator = BraintrustGenerator()

# Create a project
project = generator.generate_random_project()

# Create experiments
experiment = generator.generate_random_experiment(project["id"])
generator.generate_random_experiment_data(experiment["id"], num_records=20)

# Create datasets
dataset = generator.generate_random_dataset(project["id"])
generator.generate_random_dataset_data(dataset["id"], num_records=50)

# Generate complete setup
results = generator.generate_complete_setup(
    num_experiments=3,
    num_datasets=2,
    records_per_experiment=25,
    records_per_dataset=40
)
```

## Generated Data Types

### Sample Tasks
- text_classification
- sentiment_analysis
- question_answering
- summarization
- translation
- code_generation
- math_reasoning
- data_extraction
- chat_completion
- image_captioning

### Sample Domains
- finance, healthcare, education
- retail, technology, legal
- marketing, support, research
- entertainment

### Sample Metrics
- accuracy, precision, recall, f1_score
- bleu_score, rouge_score, semantic_similarity
- factual_accuracy, relevance, coherence
- fluency, toxicity

### Example Generated Data

**Experiment Record:**
```json
{
  "input": "Sample text for classification 1: positive excellent good neutral",
  "output": "positive",
  "expected": "positive",
  "scores": {
    "accuracy": 0.875,
    "f1_score": 0.923
  },
  "metadata": {
    "model": "gpt-4",
    "temperature": 0.7,
    "timestamp": "2025-01-15T10:30:00"
  }
}
```

**Dataset Record:**
```json
{
  "input": "Training example 5: amazing fantastic decent okay",
  "expected": "positive",
  "metadata": {
    "source": "human_annotated",
    "difficulty": "medium",
    "domain": "retail"
  }
}
```

## API Endpoints Used

- `POST /v1/project` - Create projects
- `POST /v1/experiment` - Create experiments
- `POST /v1/dataset` - Create datasets
- `POST /v1/experiment/{id}/insert` - Insert experiment data
- `POST /v1/dataset/{id}/insert` - Insert dataset data

## Output

The script creates a `braintrust_generated_data.json` file containing all generated projects, experiments, and datasets with their IDs for reference.

## Error Handling

The script includes comprehensive error handling for:
- Missing API keys
- Network connectivity issues
- API rate limits
- Invalid responses

## Rate Limiting

Built-in 1-second delays between API calls to respect Braintrust API limits.