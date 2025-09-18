#!/usr/bin/env python3
"""
Random Braintrust Experiments and Datasets Generator

This script generates random experiments and datasets using the Braintrust API.
It creates realistic test data with random parameters, inputs, and outputs.
"""

import os
import random
import string
import time
import json
from typing import List, Dict, Any
import requests
from datetime import datetime, timedelta

class BraintrustGenerator:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.environ.get("BRAINTRUST_API_KEY")
        if not self.api_key:
            raise ValueError("BRAINTRUST_API_KEY environment variable must be set")

        self.api_url = "https://api.braintrust.dev/v1"
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Sample data for generating realistic content
        self.sample_tasks = [
            "text_classification", "sentiment_analysis", "question_answering",
            "summarization", "translation", "code_generation", "math_reasoning",
            "data_extraction", "chat_completion", "image_captioning"
        ]

        self.sample_domains = [
            "finance", "healthcare", "education", "retail", "technology",
            "legal", "marketing", "support", "research", "entertainment"
        ]

        self.sample_metrics = [
            "accuracy", "precision", "recall", "f1_score", "bleu_score",
            "rouge_score", "semantic_similarity", "factual_accuracy",
            "relevance", "coherence", "fluency", "toxicity"
        ]

    def _make_request(self, method: str, endpoint: str, data: Dict = None) -> Dict:
        """Make HTTP request to Braintrust API"""
        url = f"{self.api_url}/{endpoint.lstrip('/')}"

        try:
            if method.upper() == "GET":
                response = requests.get(url, headers=self.headers, params=data)
            elif method.upper() == "POST":
                response = requests.post(url, headers=self.headers, json=data)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")

            response.raise_for_status()
            return response.json()

        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e.response, 'text'):
                print(f"Response: {e.response.text}")
            raise

    def generate_random_string(self, length: int = 8) -> str:
        """Generate random string for names and IDs"""
        return ''.join(random.choices(string.ascii_lowercase + string.digits, k=length))

    def generate_random_project(self) -> Dict:
        """Create a random project"""
        task = random.choice(self.sample_tasks)
        domain = random.choice(self.sample_domains)
        suffix = self.generate_random_string(4)

        project_name = f"{task}_{domain}_{suffix}"

        project_data = {
            "name": project_name,
            "description": f"Random {task} project for {domain} domain"
        }

        print(f"Creating project: {project_name}")
        project = self._make_request("POST", "project", project_data)
        return project

    def generate_random_experiment(self, project_id: str) -> Dict:
        """Create a random experiment within a project"""
        task = random.choice(self.sample_tasks)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = self.generate_random_string(4)

        experiment_name = f"exp_{task}_{timestamp}_{suffix}"

        experiment_data = {
            "name": experiment_name,
            "project_id": project_id,
            "description": f"Random experiment for {task} task"
        }

        print(f"Creating experiment: {experiment_name}")
        experiment = self._make_request("POST", "experiment", experiment_data)
        return experiment

    def generate_random_dataset(self, project_id: str) -> Dict:
        """Create a random dataset within a project"""
        domain = random.choice(self.sample_domains)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix = self.generate_random_string(4)

        dataset_name = f"dataset_{domain}_{timestamp}_{suffix}"

        dataset_data = {
            "name": dataset_name,
            "project_id": project_id,
            "description": f"Random dataset for {domain} domain"
        }

        print(f"Creating dataset: {dataset_name}")
        dataset = self._make_request("POST", "dataset", dataset_data)
        return dataset

    def generate_random_experiment_data(self, experiment_id: str, num_records: int = 10):
        """Generate random experiment log data"""
        records = []

        for i in range(num_records):
            # Generate random input/output pairs
            task_type = random.choice(self.sample_tasks)

            if task_type == "text_classification":
                input_text = f"Sample text for classification {i}: " + " ".join(
                    random.choices(["positive", "negative", "neutral", "good", "bad", "excellent", "poor"], k=random.randint(3, 8))
                )
                output = random.choice(["positive", "negative", "neutral"])
                expected = random.choice(["positive", "negative", "neutral"])

            elif task_type == "question_answering":
                questions = [
                    "What is the capital of France?",
                    "How does machine learning work?",
                    "What are the benefits of renewable energy?",
                    "Explain quantum computing in simple terms."
                ]
                input_text = random.choice(questions)
                output = f"Random answer {i} to the question about the topic."
                expected = f"Expected answer {i} to the question."

            else:
                input_text = f"Random input {i} for {task_type}"
                output = f"Random output {i} from model"
                expected = f"Expected output {i}"

            # Generate random scores
            scores = {}
            selected_metrics = random.sample(self.sample_metrics, random.randint(1, 3))
            for metric in selected_metrics:
                scores[metric] = round(random.uniform(0.1, 1.0), 3)

            record = {
                "input": input_text,
                "output": output,
                "expected": expected,
                "scores": scores,
                "metadata": {
                    "model": random.choice(["gpt-4", "gpt-3.5-turbo", "claude-3", "llama-2"]),
                    "temperature": round(random.uniform(0.1, 1.0), 2),
                    "timestamp": (datetime.now() - timedelta(minutes=random.randint(0, 1440))).isoformat()
                }
            }
            records.append(record)

        # Insert records into experiment
        insert_data = {"events": records}
        print(f"Inserting {num_records} records into experiment {experiment_id}")
        response = self._make_request("POST", f"experiment/{experiment_id}/insert", insert_data)
        return response

    def generate_random_dataset_data(self, dataset_id: str, num_records: int = 20):
        """Generate random dataset entries"""
        records = []

        for i in range(num_records):
            task_type = random.choice(self.sample_tasks)

            if task_type == "text_classification":
                input_text = f"Training example {i}: " + " ".join(
                    random.choices(["amazing", "terrible", "okay", "fantastic", "awful", "decent"], k=random.randint(2, 6))
                )
                expected = random.choice(["positive", "negative", "neutral"])

            elif task_type == "question_answering":
                contexts = [
                    "Paris is the capital and most populous city of France.",
                    "Machine learning is a subset of artificial intelligence.",
                    "Renewable energy comes from natural sources that are constantly replenished.",
                ]
                questions = [
                    "What is the capital of France?",
                    "What is machine learning?",
                    "What is renewable energy?",
                ]
                input_text = {
                    "context": random.choice(contexts),
                    "question": random.choice(questions)
                }
                expected = f"Ground truth answer {i}"

            else:
                input_text = f"Dataset input {i} for {task_type}"
                expected = f"Dataset expected output {i}"

            record = {
                "input": input_text,
                "expected": expected,
                "metadata": {
                    "source": random.choice(["synthetic", "human_annotated", "web_scraped"]),
                    "difficulty": random.choice(["easy", "medium", "hard"]),
                    "domain": random.choice(self.sample_domains)
                }
            }
            records.append(record)

        # Insert records into dataset
        insert_data = {"events": records}
        print(f"Inserting {num_records} records into dataset {dataset_id}")
        response = self._make_request("POST", f"dataset/{dataset_id}/insert", insert_data)
        return response

    def generate_complete_setup(self, num_experiments: int = 2, num_datasets: int = 2,
                              records_per_experiment: int = 10, records_per_dataset: int = 20):
        """Generate a complete setup with projects, experiments, and datasets"""
        print("Starting Braintrust random data generation...")

        # Create project
        project = self.generate_random_project()
        project_id = project["id"]
        print(f"Created project: {project['name']} (ID: {project_id})")

        results = {
            "project": project,
            "experiments": [],
            "datasets": []
        }

        # Create experiments
        for i in range(num_experiments):
            experiment = self.generate_random_experiment(project_id)
            experiment_id = experiment["id"]

            # Add data to experiment
            self.generate_random_experiment_data(experiment_id, records_per_experiment)
            results["experiments"].append(experiment)

            time.sleep(1)  # Rate limiting

        # Create datasets
        for i in range(num_datasets):
            dataset = self.generate_random_dataset(project_id)
            dataset_id = dataset["id"]

            # Add data to dataset
            self.generate_random_dataset_data(dataset_id, records_per_dataset)
            results["datasets"].append(dataset)

            time.sleep(1)  # Rate limiting

        print("\nGeneration complete!")
        print(f"Project: {project['name']}")
        print(f"Experiments: {[exp['name'] for exp in results['experiments']]}")
        print(f"Datasets: {[ds['name'] for ds in results['datasets']]}")

        return results


def main():
    """Main function to run the generator"""
    print("Braintrust Random Generator")
    print("=" * 40)

    try:
        generator = BraintrustGenerator()

        # Generate random setup
        results = generator.generate_complete_setup(
            num_experiments=2,
            num_datasets=2,
            records_per_experiment=15,
            records_per_dataset=25
        )

        # Save results to file
        with open("braintrust_generated_data.json", "w") as f:
            json.dump(results, f, indent=2, default=str)

        print(f"\nResults saved to: braintrust_generated_data.json")

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())