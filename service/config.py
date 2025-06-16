import logging
import os

# refer: https://stackoverflow.com/questions/384076/how-can-i-color-python-logging-output

class nfFormatter(logging.Formatter):

    grey = "\x1b[38;20m"
    yellow = "\x1b[33;20m"
    green = "\x1b[32;20m"  
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    strFormat = "[NF][%(asctime)s](%(filename)s:%(lineno)d): %(message)s "

    FORMATS = {
        logging.DEBUG:      grey + strFormat + reset,
        logging.INFO:       green + strFormat + reset,
        logging.WARNING:    yellow + strFormat + reset,
        logging.ERROR:      red + strFormat + reset,
        logging.CRITICAL:   bold_red + strFormat + reset
    }

    def format(self, record):
        logFormat = self.FORMATS.get(record.levelno)
        return logging.Formatter(logFormat).format(record)
    
# get root logger
nfLogger = logging.getLogger()
nfLogger.setLevel(logging.INFO)

consoleHandler = logging.StreamHandler()
consoleHandler.setLevel(logging.INFO)
consoleHandler.setFormatter(nfFormatter())

nfLogger.addHandler(consoleHandler)