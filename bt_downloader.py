#!/usr/bin/env python3
"""
Braintrust API Data Downloader

This script downloads experiments and datasets from a Braintrust project
and saves them as CSV files in organized directories.

Usage:
    python braintrust_downloader.py --project-id <project_id> --api-key <api_key>

Environment Variables:
    BRAINTRUST_API_KEY: Your Braintrust API key
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Any, Optional
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


class BraintrustAPIClient:
    """Client for interacting with the Braintrust API"""

    def __init__(self, api_key: str, base_url: str = "https://api.braintrust.dev"):
        self.api_key = api_key
        self.base_url = base_url
        self.session = requests.Session()

        # Set up retry strategy
        retry_strategy = Retry(
            total=3,
            status_forcelist=[429, 500, 502, 503, 504],
            backoff_factor=1
        )
        adapter = HTTPAdapter(max_retries=retry_strategy)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

        # Set headers for Braintrust API
        self.session.headers.update({
            'Authorization': f'Bearer {api_key}',
            'Content-Type': 'application/json',
            'Accept': 'application/json',
            'Accept-Encoding': 'gzip',
            'User-Agent': 'braintrust-downloader/1.0'
        })

    def _make_request(self, method: str, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """Make an authenticated request to the Braintrust API"""
        url = f"{self.base_url}/v1/{endpoint}"

        try:
            response = self.session.request(method, url, params=params)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"API request failed: {e}")
            if hasattr(e, 'response') and e.response is not None:
                print(f"Response status: {e.response.status_code}")
                print(f"Response body: {e.response.text}")
            raise

    def get_experiments(self, project_id: str) -> List[Dict]:
        """Get all experiments for a project"""
        experiments = []
        cursor = None

        while True:
            params = {'project_id': project_id, 'limit': 100}
            if cursor:
                params['cursor'] = cursor

            try:
                # Try the standard endpoint first
                response = self._make_request('GET', 'experiment', params)

                if 'objects' in response:
                    batch = response['objects']
                elif 'experiments' in response:
                    batch = response['experiments']
                elif 'data' in response:
                    batch = response['data']
                else:
                    batch = response if isinstance(response, list) else []

                experiments.extend(batch)

                # Check for pagination
                if 'cursor' in response and response['cursor']:
                    cursor = response['cursor']
                else:
                    break

            except requests.exceptions.RequestException as e:
                if "404" in str(e):
                    # Try alternative endpoint pattern
                    try:
                        response = self._make_request('GET', f'project/{project_id}/experiments', params)
                        batch = response.get('experiments', response.get('data', []))
                        experiments.extend(batch)
                        break
                    except:
                        print(f"Could not find experiments endpoint. Available experiments: {len(experiments)}")
                        break
                else:
                    raise

        return experiments

    def get_datasets(self, project_id: str) -> List[Dict]:
        """Get all datasets for a project"""
        datasets = []
        cursor = None

        while True:
            params = {'project_id': project_id, 'limit': 100}
            if cursor:
                params['cursor'] = cursor

            try:
                # Try the standard endpoint first
                response = self._make_request('GET', 'dataset', params)

                if 'objects' in response:
                    batch = response['objects']
                elif 'datasets' in response:
                    batch = response['datasets']
                elif 'data' in response:
                    batch = response['data']
                else:
                    batch = response if isinstance(response, list) else []

                datasets.extend(batch)

                # Check for pagination
                if 'cursor' in response and response['cursor']:
                    cursor = response['cursor']
                else:
                    break

            except requests.exceptions.RequestException as e:
                if "404" in str(e):
                    # Try alternative endpoint pattern
                    try:
                        response = self._make_request('GET', f'project/{project_id}/datasets', params)
                        batch = response.get('datasets', response.get('data', []))
                        datasets.extend(batch)
                        break
                    except:
                        print(f"Could not find datasets endpoint. Available datasets: {len(datasets)}")
                        break
                else:
                    raise

        return datasets

    def get_experiment_events(self, experiment_id: str) -> List[Dict]:
        """Get all events for an experiment"""
        events = []
        cursor = None

        while True:
            params = {'limit': 1000}
            if cursor:
                params['cursor'] = cursor

            try:
                response = self._make_request('GET', f'experiment/{experiment_id}/fetch', params)

                if 'objects' in response:
                    batch = response['objects']
                elif 'events' in response:
                    batch = response['events']
                elif 'data' in response:
                    batch = response['data']
                else:
                    batch = response if isinstance(response, list) else []

                events.extend(batch)

                # Check for pagination
                if 'cursor' in response and response['cursor']:
                    cursor = response['cursor']
                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Could not fetch events for experiment {experiment_id}: {e}")
                break

        return events

    def get_dataset_events(self, dataset_id: str) -> List[Dict]:
        """Get all events for a dataset"""
        events = []
        cursor = None

        while True:
            params = {'limit': 1000}
            if cursor:
                params['cursor'] = cursor

            try:
                response = self._make_request('GET', f'dataset/{dataset_id}/fetch', params)

                if 'objects' in response:
                    batch = response['objects']
                elif 'events' in response:
                    batch = response['events']
                elif 'data' in response:
                    batch = response['data']
                else:
                    batch = response if isinstance(response, list) else []

                events.extend(batch)

                # Check for pagination
                if 'cursor' in response and response['cursor']:
                    cursor = response['cursor']
                else:
                    break

            except requests.exceptions.RequestException as e:
                print(f"Could not fetch events for dataset {dataset_id}: {e}")
                break

        return events


def flatten_dict(d: Dict, parent_key: str = '', sep: str = '_') -> Dict:
    """Flatten a nested dictionary for CSV export"""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        elif isinstance(v, list):
            # Convert lists to JSON strings
            items.append((new_key, json.dumps(v)))
        else:
            items.append((new_key, v))
    return dict(items)


def save_to_csv(data: List[Dict], filepath: Path) -> None:
    """Save a list of dictionaries to a CSV file"""
    if not data:
        print(f"No data to save for {filepath}")
        return

    # Flatten nested dictionaries
    flattened_data = [flatten_dict(item) for item in data]

    # Get all possible fieldnames
    fieldnames = set()
    for item in flattened_data:
        fieldnames.update(item.keys())

    fieldnames = sorted(list(fieldnames))

    # Create directory if it doesn't exist
    filepath.parent.mkdir(parents=True, exist_ok=True)

    # Write CSV
    with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(flattened_data)

    print(f"Saved {len(data)} records to {filepath}")


def download_braintrust_data(project_id: str, api_key: str, output_dir: str = "braintrust_data") -> None:
    """Download all experiments and datasets from a Braintrust project"""

    client = BraintrustAPIClient(api_key)

    # Create project-specific directory within output_dir
    output_path = Path(output_dir) / project_id

    print(f"Downloading data for project: {project_id}")
    print(f"Output directory: {output_path.absolute()}")

    # Create main output directory
    output_path.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    experiments_dir = output_path / "experiments"
    datasets_dir = output_path / "datasets"
    experiments_dir.mkdir(exist_ok=True)
    datasets_dir.mkdir(exist_ok=True)

    try:
        # Download experiments
        print("\nFetching experiments...")
        experiments = client.get_experiments(project_id)
        print(f"Found {len(experiments)} experiments")

        for experiment in experiments:
            exp_id = experiment.get('id', 'unknown')
            exp_name = experiment.get('name', f'experiment_{exp_id}')

            # Sanitize filename
            safe_name = "".join(c for c in exp_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_{exp_id}.csv"

            print(f"Processing experiment: {exp_name} ({exp_id})")

            # Get experiment metadata and events
            events = client.get_experiment_events(exp_id)

            # Combine metadata with events
            exp_data = []
            if events:
                # Add experiment metadata to each event
                for event in events:
                    combined_record = {
                        'experiment_id': exp_id,
                        'experiment_name': exp_name,
                        **experiment,
                        **event
                    }
                    exp_data.append(combined_record)
            else:
                # If no events, save just the experiment metadata
                exp_data = [experiment]

            # Save to CSV
            filepath = experiments_dir / filename
            save_to_csv(exp_data, filepath)

        # Download datasets
        print("\nFetching datasets...")
        datasets = client.get_datasets(project_id)
        print(f"Found {len(datasets)} datasets")

        for dataset in datasets:
            ds_id = dataset.get('id', 'unknown')
            ds_name = dataset.get('name', f'dataset_{ds_id}')

            # Sanitize filename
            safe_name = "".join(c for c in ds_name if c.isalnum() or c in (' ', '-', '_')).rstrip()
            filename = f"{safe_name}_{ds_id}.csv"

            print(f"Processing dataset: {ds_name} ({ds_id})")

            # Get dataset metadata and events
            events = client.get_dataset_events(ds_id)

            # Combine metadata with events
            ds_data = []
            if events:
                # Add dataset metadata to each event
                for event in events:
                    combined_record = {
                        'dataset_id': ds_id,
                        'dataset_name': ds_name,
                        **dataset,
                        **event
                    }
                    ds_data.append(combined_record)
            else:
                # If no events, save just the dataset metadata
                ds_data = [dataset]

            # Save to CSV
            filepath = datasets_dir / filename
            save_to_csv(ds_data, filepath)

        print(f"\nDownload completed! Data saved to: {output_path.absolute()}")
        print(f"- Experiments: {len(experiments)} files in {experiments_dir}")
        print(f"- Datasets: {len(datasets)} files in {datasets_dir}")

    except Exception as e:
        print(f"Error downloading data: {e}")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Download experiments and datasets from Braintrust API",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python braintrust_downloader.py --project-id proj_abc123 --api-key sk_abc123
  python braintrust_downloader.py --project-id proj_abc123 --output-dir ./my_data

  # Using environment variable for API key
  export BRAINTRUST_API_KEY=sk_abc123
  python braintrust_downloader.py --project-id proj_abc123
        """
    )

    parser.add_argument(
        '--project-id',
        required=True,
        help='Braintrust project ID'
    )

    parser.add_argument(
        '--api-key',
        help='Braintrust API key (or set BRAINTRUST_API_KEY environment variable)'
    )

    parser.add_argument(
        '--output-dir',
        default='braintrust_data',
        help='Output directory for CSV files (default: braintrust_data)'
    )

    args = parser.parse_args()

    # Get API key from args or environment
    api_key = args.api_key or os.environ.get('BRAINTRUST_API_KEY')
    if not api_key:
        print("Error: API key is required. Provide it via --api-key or BRAINTRUST_API_KEY environment variable.")
        sys.exit(1)

    download_braintrust_data(args.project_id, api_key, args.output_dir)


if __name__ == '__main__':
    main()