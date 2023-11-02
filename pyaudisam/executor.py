# coding: utf-8

# PyAuDiSam: Automation of Distance Sampling analyses with Distance software (http://distancesampling.org/)

# Copyright (C) 2021 Jean-Philippe Meuret

# This program is free software: you can redistribute it and/or modify it under the terms
# of the GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or (at your option) any later version.
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
# without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the GNU General Public License for more details.
# You should have received a copy of the GNU General Public License along with this program.
# If not, see https://www.gnu.org/licenses/.

# Submodule "executor": Tools for easily running analyses sequentially or parallely

import sys
import os
import concurrent.futures as cofu
import warnings

from . import log

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


class SequentialExecutor(cofu.Executor):

    """Non-parallel concurrent.futures.Executor minimal implementation
    """

    def __init__(self):
        
        logger.info2('Started the SequentialExecutor.')
            
    def submit(self, func, *args, **kwargs):
        
        return ImmediateFuture(func(*args, **kwargs))  # Do it now !
    
    def map(self, func, *iterables, timeout=None, chunksize=1):
        
        return map(func, *iterables)  # Do it now !
    
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
        :param threads: Must be None or >= 0 ; 0 for auto-number (see expectedWorkers function) ;
                        None or 1 for no actual parallelism = sequential execution,
                        but 1 is slightly different, in that it means asynchronous call,
                        whereas None means pure sequential calling ;
                        if processes is not None, must be None (= unspecified)
        :param processes: Must be None or >= 0 ; 0 for auto-number (see expectedWorkers function), ;
                          None or 1 for no actual parallelism = sequential execution,
                          but 1 is slightly different, in that it means asynchronous call,
                          whereas None means pure sequential calling ;,
                          if threads is not None, must be None (= unspecified)
        :param name_prefix: See concurrent module (only for multi-threading)
        :param mp_context: See concurrent module (only for multiprocessing)
        :param initializer: See concurrent module
        :param initargs: See concurrent module
        """
        
        assert (threads is None and (processes is None or processes >= 0)) \
               or (processes is None and (threads is None or threads >= 0)), \
               'An Executor can\'t implement multi-threading _and_ multi-processing at the same time'
               
        # Keep original parallelism (or not) specs for expectedWorkers().
        self.threads = threads
        self.processes = processes

        # Create / Get the actual executor object.
        self.realExor = None

        if threads is not None:
            self.realExor = \
                cofu.ThreadPoolExecutor(max_workers=threads or None,
                                        thread_name_prefix=name_prefix,
                                        initializer=None, initargs=initargs)
            logger.info1('Started a ThreadPoolExecutor(max_workers={})'.format(threads or 'None'))
        
        elif processes is not None:
            self.realExor = \
                cofu.ProcessPoolExecutor(max_workers=processes or None,
                                         mp_context=mp_context,
                                         initializer=initializer, initargs=initargs)
            logger.info1('Started a ProcessPoolExecutor(max_workers={})'.format(processes or 'None'))
                    
        else:
            if self.TheSeqExor is None:
                self.TheSeqExor = SequentialExecutor()
            self.realExor = self.TheSeqExor
    
    def expectedWorkers(self):

        """Compute the theoretically expected to be used number of thread/process workers
        of an Executor instance, from the specified number of threads / processes

        Warning: Fully reports what's in the actual implementation of concurrent.futures.ThreadPoolExecutor
                 and concurrent.futures.ProcessPoolExecutor : figures are verified only for python versions
                 from 3.0 to 3.10pre
        """

        if sys.version_info.major < 3 or sys.version_info.minor < 5 or sys.version_info.minor > 10:
            warnings.warn('Executor.expectedWorkers() may not report accurate figures as Python < 3.5 or > 3.10',
                          RuntimeWarning)

        if self.threads is None:
            if self.processes is None:
                return 1
            elif self.processes == 0:
                return os.cpu_count()
            else:
                return self.processes
        elif self.threads == 0:
            return 5 * os.cpu_count() if sys.version_info.minor < 8 else min(32, os.cpu_count() + 4)
        else:
            return self.threads

    def isParallel(self):
    
        return self.realExor is not self.TheSeqExor and self.realExor._max_workers > 1
    
    def isAsync(self):
    
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
            logger.info2(self.realExor.__class__.__name__ + ' shut down.')
            self.realExor.shutdown(wait=wait)
        self.realExor = None
        
#    def __del__(self):
#    
#        self.shutdown()
            

if __name__ == '__main__':

    sys.exit(0)
