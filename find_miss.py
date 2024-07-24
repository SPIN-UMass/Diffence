import os
import re

def check_metrics(file_path):
    try:
        with open(file_path, 'r') as file:
            content = file.read()
            has_auc = re.search(r'Best Attack AUC:\s*([0.0-9.]+)', content) is not None
            has_acc = re.search(r'Best Attack ACC:\s*([0.0-9.]+)', content) is not None
            return has_auc, has_acc
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return False, False

def identify_missing_metrics(base_dir):
    missing_files = []
    for root, dirs, files in os.walk(base_dir):
        for file_name in files:
            if file_name.endswith('w_Diffence') or file_name.endswith('undefended'):
                file_path = os.path.join(root, file_name)
                has_auc, has_acc = check_metrics(file_path)
                if not has_auc or not has_acc:
                    missing_files.append((file_path, has_auc, has_acc))
                    print(f"File missing metrics: {file_path}, AUC found: {has_auc}, ACC found: {has_acc}")
    return missing_files

# Example usage
base_dir = './results'
missing_files = identify_missing_metrics(base_dir)

# Save the list of missing files to a text file
output_path = './missing_metrics_files.txt'
with open(output_path, 'w') as output_file:
    for file_path, has_auc, has_acc in missing_files:
        output_file.write(f"{file_path}, AUC found: {has_auc}, ACC found: {has_acc}\n")

print(f"List of files with missing metrics saved to {output_path}")
