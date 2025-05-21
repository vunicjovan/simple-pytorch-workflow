import logging
import os
from pathlib import Path


class LogConfiguration:
    """
    Configures the basic logging settings for the application.
    """

    @staticmethod
    def configure_logging(file_name: str) -> logging.Logger:
        """
        Applies basic logging configuration to be used across the application.
        Returns a logger instance.

        Parameters
        ----------
        file_name : str
            Name of the file for which logger instance is being obtained

        Returns
        -------
        logging.Logger
            Instance of logger for the given file
        """

        logs_dir = "logs"
        log_file_path = os.path.join(logs_dir, "main.log")

        Path(logs_dir).mkdir(parents=True, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(filename=log_file_path, mode="w"),
                logging.StreamHandler(),
            ],
        )

        return logging.getLogger(file_name)
