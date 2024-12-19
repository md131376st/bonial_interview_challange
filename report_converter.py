import os
import subprocess

# Paths
input_dir = "notebooks"
output_dir = "../reports/churn prediction"


os.makedirs(output_dir, exist_ok=True)


for file in os.listdir(input_dir):
    if file.endswith(".ipynb"):
        input_path = os.path.join(input_dir, file)
        output_name = os.path.splitext(file)[0]  # Remove the .ipynb extension
        output_path = os.path.join(output_dir, output_name)
        # Run nbconvert
        subprocess.run([
            "jupyter", "nbconvert",
            "--to", "markdown",
            input_path,
            "--output", output_path
        ])
print("Conversion completed!")