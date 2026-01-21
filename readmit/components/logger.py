import logging
import os, sys
from datetime import datetime


LOGS_DIR = "logs"
os.makedirs(LOGS_DIR, exist_ok=True)

LOG_FILE = os.path.join(LOGS_DIR, f"log_{datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}.log")
# LOG_PATH = os.path.join(LOG_DIR, LOG_FILE)

# Define handlers
handlers = [
    logging.StreamHandler(sys.stdout),   # Console output
    logging.FileHandler(LOG_FILE, encoding='utf-8')        # File output
]

logging.basicConfig(
    # filename=LOG_FILE,
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    handlers=handlers  # Specify handlers directly
)

logging.getLogger("matplotlib").setLevel(logging.WARNING)
logging.getLogger("sklearn").setLevel(logging.WARNING)

# Optional test
# if __name__ == "__main__":
#     logging.info("✅ Logging setup successful.")
#     logging.info("This is an info message")
#     logging.warning("This is a warning message")