import os
from tkinter import filedialog


def check_requirements(requirements_file="ATK/requirements.txt"):
    with open(requirements_file) as f:
        missing = [pkg.split("==")[0] for pkg in f if pkg.strip() and not pkg.startswith("#") and not __import__(pkg.split("==")[0], globals(), locals(), [], 0)] # noqa:
    if missing:
        print(f"Error: Missing packages: {missing}\nPlease install with:\n    pip install -r {requirements_file}") # noqa:
        os._exit(1)
    print("All required packages are installed.")


# Modified select_file function
def select_file(file_name_var):
    file_path = filedialog.askopenfilename(
        initialdir=os.getcwd(),
        title="Select a CSV file",
        filetypes=(("CSV files", "*.csv"), ("All files", "*.*"))
    )
    if file_path:
        file_name_var.set(os.path.basename(file_path))  # Update the displayed filename # noqa:
        return file_path  # Return the file path to the caller
    return None
