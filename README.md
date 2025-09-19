# Braintrust Data Generator and Downloader

This repository contains two Python scripts for working with the Braintrust API data:

1. **`data_generator.py`** - Generates random experiments and datasets for testing
2. **`bt_downloader.py`** - Downloads experiments and datasets from Braintrust projects

## Prerequisites

- Python 3.6+
- Braintrust API key
- Required packages: `requests`

```bash
pip3 install requests
```

## Setup

Set your Braintrust API key is needed as an environment variable:

```bash
export BRAINTRUST_API_KEY=your_api_key_here
```

## Usage

### Step 1: Generate Test Data

Use `data_generator.py` to create random experiments and datasets for testing:

```bash
python3 data_generator.py
```

**What it does:**
- Creates a new random project
- Generates 2 experiments with 15 records each
- Generates 2 datasets with 25 records each
- Saves project details to `braintrust_generated_data.json`

**Generated data includes:**
- Text classification tasks
- Question answering examples
- Sentiment analysis samples
- Random scores and metadata
- Various AI models and parameters

### Step 2: Download the Generated Data

Use `bt_downloader.py` to download and export the generated data as CSV files:

```bash
python bt_downloader.py --project-id <project_id>
```

These CSV files will be saved in the directory the `bt_downloader.py` script is ran from by default, in the `braintrust_data` directory. Inside the `braintrust_data` directory the data will then be stored in a directory named after the `project_id`.

**Command Options:**
- `--project-id` (required): The Braintrust project ID to download
- `--api-key` (optional): API key if not set as environment variable
- `--output-dir` (optional): Output directory (default: `braintrust_data`)

**Examples:**
```bash
# Basic usage
python bt_downloader.py --project-id proj_abc123

# Specify custom output directory
python bt_downloader.py --project-id proj_abc123 --output-dir ./my_data

# Include API key in command
python bt_downloader.py --project-id proj_abc123 --api-key sk_abc123
```

## Output Structure

The downloader creates the following directory structure:

```
braintrust_data/
└── <project_id>/
    ├── experiments/
    │   ├── exp_text_classification_20241201_120000_a1b2.csv
    │   └── exp_sentiment_analysis_20241201_120100_c3d4.csv
    └── datasets/
        ├── dataset_finance_20241201_120200_e5f6.csv
        └── dataset_healthcare_20241201_120300_g7h8.csv
```

Each CSV file contains:
- **Experiments**: Input/output data, scores, metadata, and experiment details
- **Datasets**: Training examples, expected outputs, and dataset metadata

## Workflow Example

1. **Generate test data:**
   ```bash
   python data_generator.py
   ```

2. **Check the generated project ID:**
   ```bash
   cat braintrust_generated_data.json | grep '"id"'
   ```

3. **Download the data:**
   ```bash
   python bt_downloader.py --project-id <project_id_from_step_2>
   ```

4. **Find your CSV files:**
   ```bash
   ls -la braintrust_data/<project_id>/
   ```

## Features

### Data Generator (`data_generator.py`)
- Creates realistic AI/ML test scenarios
- Supports multiple task types (classification, QA, summarization, etc.)
- Generates random but coherent data
- Includes metadata and scoring metrics
- Rate limiting to respect API limits

### Data Downloader (`bt_downloader.py`)
- Handles pagination for large datasets
- Flattens nested JSON structures for CSV export
- Robust error handling and retry logic
- Sanitizes filenames for cross-platform compatibility
- Preserves all metadata and relationships

## Error Handling

Both scripts include comprehensive error handling:
- API authentication validation
- Network retry logic with backoff
- Graceful handling of missing data
- Clear error messages and status updates

## Customization

You can modify the generation parameters in `data_generator.py`:

```python
# In the main() function
results = generator.generate_complete_setup(
    num_experiments=5,        # Number of experiments
    num_datasets=3,           # Number of datasets
    records_per_experiment=50, # Records per experiment
    records_per_dataset=100   # Records per dataset
)
```

## Troubleshooting

**API Key Issues:**
```bash
# Verify your API key is set
echo $BRAINTRUST_API_KEY
```

**Permission Errors:**
- Ensure your API key has access to the specified project
- Check that the project ID is correct

**Network Issues:**
- Both scripts include retry logic for temporary failures
- Check your internet connection and API endpoint accessibility