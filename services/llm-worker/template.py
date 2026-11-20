import os
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s - %(levelname)s - %(message)s")

list_of_files = [
    ".env",
    "requirements.txt",
    "src/__init__.py",
    "src/prompts.py",
    "src/helpers.py",
    "setup.py",
    "app.py",
    "research/notebook.ipynb",
]

for file_path in list_of_files:
    file_path = Path(file_path)
    file_dir, file_name = os.path.split(file_path)

    if file_dir != "":
        os.makedirs(file_dir, exist_ok=True)
        logging.info(f"Created directory: {file_dir}")

    if (not os.path.exists(file_path)) or (os.path.getsize(file_path) == 0):
        with open(file_path, "w") as f:
            pass
            logging.info(f"Created empty file: {file_path}")
    else:
        logging.info(f"Skipping file: {file_path} (already exists)")

logging.info("Project structure created successfully.")
