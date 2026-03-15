# setup_environment.py

import subprocess
import sys
import os
import pkg_resources # To check installed package versions
import nltk # To handle NLTK data downloads

def check_and_install_packages():
    """
    Checks for required Python packages and installs them if missing or
    if their versions don't match the specified ones.
    """
    required_packages = {
        "polars": "1.31.0",
        "scikit-learn": "1.7.0",
        "gensim": "4.3.0",
        "nltk": "3.8.1",
        "xgboost": "3.0.2"
    }

    print("--- Checking and Installing Python Packages ---")
    for package, required_version in required_packages.items():
        try:
            # Check if package is installed and get its version
            installed_version = pkg_resources.get_distribution(package).version
            print(f"  {package}: Installed version {installed_version}, Required version {required_version}")

            # Check if installed version matches required version
            if installed_version != required_version:
                print(f"  {package}: Version mismatch. Attempting to upgrade/downgrade to {required_version}...")
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={required_version}"])
                print(f"  {package}: Successfully installed version {required_version}.")
            else:
                print(f"  {package}: Version matches. OK.")

        except pkg_resources.DistributionNotFound:
            print(f"  {package}: Not found. Attempting to install version {required_version}...")
            try:
                subprocess.check_call([sys.executable, "-m", "pip", "install", f"{package}=={required_version}"])
                print(f"  {package}: Successfully installed version {required_version}.")
            except subprocess.CalledProcessError as e:
                print(f"  Error installing {package}: {e}")
                sys.exit(1) # Exit if a critical package fails to install

        except subprocess.CalledProcessError as e:
            print(f"  Error checking/installing {package}: {e}")
            sys.exit(1)

    print("--- All Python packages checked/installed. ---")

def download_nltk_data():
    """
    Downloads required NLTK data if not already present.
    """
    print("\n--- Checking and Downloading NLTK Data ---")
    datasets = ['punkt', 'stopwords', 'wordnet', 'omw-1.4']

    for dataset in datasets:
        try:
            # NLTK data paths vary, 'punkt' is a tokenizer, others are corpora.
            if dataset == 'punkt':
                nltk.data.find(f'tokenizers/{dataset}')
            elif dataset == 'omw-1.4':
                nltk.data.find(f'corpora/{dataset}')
            else:
                nltk.data.find(f'corpora/{dataset}')
            print(f"  NLTK data '{dataset}' already present. OK.")
        except Exception: # Catching a general Exception is robust for nltk.data.find()
            print(f"  NLTK data '{dataset}' not found. Downloading...")
            try:
                nltk.download(dataset)
                print(f"  NLTK data '{dataset}' downloaded successfully.")
            except Exception as e:
                print(f"  Error downloading NLTK data '{dataset}': {e}")
                print("  Please try to download manually if issues persist (e.g., in Python: import nltk; nltk.download('all')).")

    print("--- All NLTK data checked/downloaded. ---")

def main():
    """Main function to run the environment setup."""
    print("--- Starting Environment Setup Script ---")

    # Optional: Check for virtual environment
    if hasattr(sys, 'real_prefix') or (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("  Running inside a virtual environment. Good!")
    else:
        print("  WARNING: Not running inside a virtual environment. It's highly recommended to use one.")
        print("  Consider creating and activating a virtual environment (e.g., 'conda create -n myenv python=3.11' then 'conda activate myenv')")
        # You might add sys.exit(1) here if you want to force virtual env usage.

    # Check and install Python packages
    check_and_install_packages()

    # Download NLTK data
    download_nltk_data()

    print("\n--- Environment Setup Complete. You are ready to run your project! ---")
    print(f"Python version: {sys.version}")

if __name__ == "__main__":
    main()