# coding: utf-8

# Thin wrapper above logging to get more debug and info levels
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

import sys
import logging
from logging import NOTSET, DEBUG, INFO, WARNING, ERROR, CRITICAL


# Define new logging levels.
DEBUG0 = DEBUG
logging.addLevelName(DEBUG0, "DEBUG0")
DEBUG1 = DEBUG - 1
logging.addLevelName(DEBUG1, "DEBUG1")
DEBUG2 = DEBUG - 2
logging.addLevelName(DEBUG2, "DEBUG2")
DEBUG3 = DEBUG - 3
logging.addLevelName(DEBUG3, "DEBUG3")
DEBUG4 = DEBUG - 4
logging.addLevelName(DEBUG4, "DEBUG4")

INFO0 = INFO
logging.addLevelName(INFO0, "INFO0")
INFO1 = INFO - 1
logging.addLevelName(INFO1, "INFO1")
INFO2 = INFO - 2
logging.addLevelName(INFO2, "INFO2")
INFO3 = INFO - 3
logging.addLevelName(INFO3, "INFO3")
INFO4 = INFO - 4
logging.addLevelName(INFO4, "INFO4")


class Logger(logging.Logger):

    """A Logger class with methods associated to the new levels
    """

    Configured = False

    def __init__(self, name):
        super().__init__(name)

    info0 = logging.Logger.info
    
    def info1(self, msg, *args, **kwargs):
        self.log(INFO1, msg, *args, **kwargs)

    def info2(self, msg, *args, **kwargs):
        self.log(INFO2, msg, *args, **kwargs)

    def info3(self, msg, *args, **kwargs):
        self.log(INFO3, msg, *args, **kwargs)

    def info4(self, msg, *args, **kwargs):
        self.log(INFO4, msg, *args, **kwargs)

    debug0 = logging.Logger.debug
    
    def debug1(self, msg, *args, **kwargs):
        self.log(DEBUG1, msg, *args, **kwargs)

    def debug2(self, msg, *args, **kwargs):
        self.log(DEBUG2, msg, *args, **kwargs)

    def debug3(self, msg, *args, **kwargs):
        self.log(DEBUG3, msg, *args, **kwargs)

    def debug4(self, msg, *args, **kwargs):
        self.log(DEBUG4, msg, *args, **kwargs)

    @staticmethod
    def configure(level=NOTSET, handlers=[sys.stdout], fileMode='w', verbose=False,
                  format='%(asctime)s %(name)s %(levelname)s\t%(message)s', reset=False):
        
        """Configure logging system, mainly the root logger (levels, handlers, formatter, ...)

        Parameters:
        :param level: see logging.Logger.setLevel
        :param handlers: a list of "handler specs" ; according to type,
            * str: logging.FileHandler for given file path-name
            * otherwise: StreamHandler (for sys.stdout and so on)
            * None or empty list => use currently configured ones for root logger
        :param fileMode: see logging.FileHandler ctor
        :param format: see logging.Handler.setFormatter
        :param verbose: if True, write a first INFO msg to the handlers' targets
        :param reset: if True, hard cleanup logger config. (useful in jupyter notebooks)
        """

        # Configure root logger (assuming children have propagate=on).
        # Note: Setting handlers for multiple children rather than once and for all for root ...
        #        gives bad things on FileHandlers, with many missing / intermixed / unsorted lines ...
        #        => unusable. Whereas it seems to work well with StreamHandlers
        root = logging.getLogger()

        if reset:
            while root.handlers:
                root.handlers.pop()
     
        formatter = logging.Formatter(format)
        for hdlr in handlers:
            if isinstance(hdlr, str):
                handler = logging.FileHandler(hdlr, mode=fileMode)
            else:
                handler = logging.StreamHandler(stream=hdlr)
            handler.setFormatter(formatter)
            root.addHandler(handler)
        
        if verbose:
            root.setLevel(INFO)
            def handlerId(hdlr):
                return 'File({})'.format(hdlr) if isinstance(hdlr, str) else 'Stream({})'.format(hdlr.name)
            root.info('Will log to {}'.format(', '.join(handlerId(hdlr) for hdlr in handlers)))

        if not verbose or level != INFO:
            root.setLevel(level)

        Logger.Configured = True

    @staticmethod
    def logger(name, level=NOTSET, reset=False):

        """ Create and setup the logger with given name.
        
        Parameters:
        :param name: see logging.getLogger
        :param level: see logging.Logger.setLevel
        :param verbose: if True, write a first INFO msg to the handlers' targets
        :param reset: if True, hard cleanup logger config. (useful in jupyter notebooks)
        """
        
        if not Logger.Configured:
            Logger.configure(level=INFO, reset=reset)

        # Create / get logger.
        logr = logging.getLogger(name)
        
        # Cleanup any default handler if any (ex: jupyter does some logging initialisation itself ...)
        if reset:
            while logr.handlers:
                logr.handlers.pop()
        
        # Set level
        logr.setLevel(level)
        
        return logr


logging.setLoggerClass(Logger)

configure = Logger.configure

logger = Logger.logger
