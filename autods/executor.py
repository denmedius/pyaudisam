# coding: utf-8

# Automation of Distance Sampling analyses with Distance software
#  http://distancesampling.org/
#
# Executor: Tools for easily running analyses sequentially or parallely
#
# Author: Jean-Philippe Meuret (http://jpmeuret.free.fr/)
# License: GPL 3

import concurrent.futures as cofu

import autods.log as log

logger = log.logger('ads.exr')


class ImmediateFuture(object):
    
    """Synchronous concurrent.futures.Future minimal and trivial implementation,
       for use with SequentialExecutor
    """

    def __init__(self, result):
        
        self._result = result
            
    def result(self, timeout=None):
        
        return self._result

    def exception(self, timeout=None):
        
        return None

    def cancel(self):
        
        return False

    def cancelled(self):
        
        return False

    def running(self):
        
        return False

    def done(self):
        
        return True

    def exception(self):
        
        return None


class SequentialExecutor(cofu.Executor):

    """Non-parallel concurrent.futures.Executor minimal implementation
    """

    def __init__(self):
        
        logger.debug('Started the SequentialExecutor.')
            
    def submit(self, func, *args, **kwargs):
        
        return ImmediateFuture(func(*args, **kwargs)) # Do it now !
    
    def map(self, func, *iterables, timeout=None, chunksize=1):
        
        return map(func, *iterables) # Do it now !
    
    def shutdown(self, wait=True):
        
        pass
        
        
class Executor(object):

    """Wrapper class for simpler concurrent.futures.Executor interface,
       and access to added non-parallel SequentialExecutor
    """

    # The only SequentialExecutor (only one needed)
    TheSeqExor = None

    def __init__(self, threads=None, processes=None,
                 name_prefix='', mp_context=None, initializer=None, initargs=()):

        """Ctor
        
        Parameters:
        :param threads: Must be None or >= 0 ; 0 for auto-number (5 x nb of actual CPUs) ;
                        None or 1 for no actual parallelism = sequential execution,
                        if processes is not None, must be None (= unspecified)
        :param processes: Must be None or >= 0 ; 0 for auto-number (nb of actual CPUs), ;
                          None or 1 for no actual parallelism = sequential execution,
                          if threads is not None, must be None (= unspecified)
        :param name_prefix: See concurrent module (only for multi-threading)
        :param mp_context: See concurrent module (only for multi-processing)
        :param initializer: See concurrent module
        :param initargs: See concurrent module
        """
        
        assert (threads is None and (processes is None or processes >= 0)) \
               or (processes is None and (threads is None or threads >= 0)), \
               'An Executor can\'t implement multi-threading _and_ multi-processing at the same time'
               
        self.realExor = None

        if not(threads is None or threads == 1):
            self.realExor = \
                cofu.ThreadPoolExecutor(max_workers=threads or None,
                                        thread_name_prefix=name_prefix,
                                        initializer=None, initargs=initargs)
            logger.debug('Started a ThreadPoolExecutor(max_workers={})'.format(threads or 'None'))
        
        elif not(processes is None or processes == 1):
            self.realExor = \
                cofu.ProcessPoolExecutor(max_workers=processes or None,
                                         mp_context=mp_context,
                                         initializer=initializer, initargs=initargs)
            logger.debug('Started a ProcessPoolExecutor(max_workers={})'.format(processes or 'None'))
                    
        else:
            if self.TheSeqExor is None:
                self.TheSeqExor = SequentialExecutor()
            self.realExor = self.TheSeqExor
    
    def isParallel(self):
    
        return self.realExor is not self.TheSeqExor
    
    def submit(self, func, *args, **kwargs):
    
        assert self.realExor is not None, 'Can\'t submit after shutdown'
        
        return self.realExor.submit(func, *args, **kwargs)
    
    def map(self, func, *iterables, timeout=None, chunksize=1):
        
        return self.realExor.map(func, *iterables, timeout=timeout, chunksize=chunksize)
        
    def asCompleted(self, futures):
    
        return iter(futures) if isinstance(self.realExor, SequentialExecutor) \
               else cofu.as_completed(futures)
    
    def shutdown(self, wait=True):
              
        if self.realExor is not None and self.realExor is not self.TheSeqExor:
            logger.debug(self.realExor.__class__.__name__ + ' shut down.')
            self.realExor.shutdown(wait=wait)
        self.realExor = None
        
#    def __del__(self):
#    
#        self.shutdown()
            

if __name__ == '__main__':

    sys.exit(0)
