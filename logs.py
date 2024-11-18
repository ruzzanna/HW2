import logging


class EnhancedFormatter(logging.Formatter):
    """
    A custom logging formatter to add color coding to log messages based on their severity level.
    """

    # ANSI color codes
    white = "\033[37m"
    blue = "\033[34m"
    yellow = "\033[33m"
    red = "\033[31m"
    bold_red = "\033[1;31m"
    reset = "\033[0m"

    log_format = "%(asctime)s | %(name)s | %(levelname)s: %(message)s (at line %(lineno)d)"

    level_colors = {
        logging.DEBUG: white,
        logging.INFO: blue,
        logging.WARNING: yellow,
        logging.ERROR: red,
        logging.CRITICAL: bold_red,
    }

    def format(self, record):
        """
        Apply color coding to the log message based on its severity level.
        """
        color = self.level_colors.get(record.levelno, self.reset)

        colored_format = color + self.log_format + self.reset

        formatter = logging.Formatter(colored_format)
        return formatter.format(record)