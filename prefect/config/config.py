from pathlib import Path

# Directories
BASE_DIR = Path(__file__).parent.parent.absolute()
CONFIG_DIR = Path(BASE_DIR, "config")
DATA_DIR = Path((BASE_DIR).parent.absolute(), "data")
LOGS_DIR = Path(BASE_DIR, "logs")
LOGS_DIR.mkdir(parents=True, exist_ok=True)

import logging
import sys

from rich.logging import RichHandler

logging.basicConfig(
    level="NOTSET",
    format="%(message)s",
    datefmt="[%X]",
    handlers=[RichHandler(rich_tracebacks=True)],
)
logger = logging.getLogger()

if __name__ == "__main__":
    logging.debug("Used for debugging your code.")
    logging.info("Informative messages from your code.")
    logging.warning("Everything works but there is something to be aware of.")
    logging.error("There's been a mistake with the process.")
    logging.critical("There is something terribly wrong and process may terminate.")
