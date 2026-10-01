import os
from pathlib import Path

# Directory paths sent to /ingest/ must resolve inside this root.
DATA_DIR = Path(os.getenv("DATA_DIR", "/data"))

MAX_FILE_SIZE_MB = int(os.getenv("MAX_FILE_SIZE_MB", "50"))
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024
MAX_FILES_PER_REQUEST = int(os.getenv("MAX_FILES_PER_REQUEST", "20"))
