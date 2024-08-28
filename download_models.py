import os
import gdown
import argparse
import shutil

# Mapping of datasets to their corresponding Google Drive URLs
URLS = {
    'cifar10': {
        'diffusion': 'https://drive.google.com/drive/folders/14CtR6amKoKd2GEs-gaIgowJlyvD6JqsA?usp=sharing',
        'target': 'https://drive.google.com/drive/folders/1hsEBCOjZTPGQCgc_gYqXPcbfyc4ZelD0?usp=sharing'
    },
    'cifar100': {
        'diffusion': 'https://drive.google.com/drive/folders/1jBYV3gmUhKHJbUQ32cFAJSlouhzGTVTh?usp=sharing',
        'target': 'https://drive.google.com/drive/folders/1skKxqe8X2WZqOVLJYlGENzPQ8j-F5IC7?usp=sharing'
    },
    'svhn': {
        'diffusion': 'https://drive.google.com/drive/folders/1AqvHjCbYTDQ3hcuPSOxxlynvASx3hdua?usp=sharing',
        'target': 'https://drive.google.com/drive/folders/1oh8coO5Qh79fUktSYCYFlmInZlsmUEjh?usp=sharing'
    }
}

def download_folder(url, output_dir):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Download the folder contents using gdown
    gdown.download_folder(url, output=output_dir, quiet=False, use_cookies=False)

    # Flatten the directory structure if nested directories are created
    flatten_directory(output_dir)

def flatten_directory(output_dir):
    for root, dirs, files in os.walk(output_dir):
        for dir_name in dirs:
            dir_path = os.path.join(root, dir_name)
            # Move files from nested directories to the main output directory
            for file_name in os.listdir(dir_path):
                src_file = os.path.join(dir_path, file_name)
                dst_file = os.path.join(output_dir, file_name)
                if os.path.isfile(src_file):
                    print(f"Moving {src_file} to {dst_file}")
                    shutil.move(src_file, dst_file)
            # After moving, remove the now-empty directory
            shutil.rmtree(dir_path)

def download_models(dataset):
    # Check if the dataset is valid
    if dataset not in URLS:
        print(f"Dataset {dataset} is not recognized. Please choose from {list(URLS.keys())}.")
        return

    # Define output directories based on the dataset argument
    diffusion_output_dir = os.path.join(dataset, 'diff_defense', 'diff_models')
    target_output_dir = os.path.join(dataset, 'final-all-models', 'resnet')

    # Download diffusion models
    print(f"Downloading diffusion models for {dataset}...")
    download_folder(URLS[dataset]['diffusion'], diffusion_output_dir)

    # Download target models
    print(f"Downloading target models for {dataset}...")
    download_folder(URLS[dataset]['target'], target_output_dir)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download pretrained models for a specific dataset.")
    parser.add_argument('--dataset', type=str, required=True, help='Name of the dataset (e.g., cifar10, cifar100, svhn)')
    args = parser.parse_args()

    download_models(args.dataset)
