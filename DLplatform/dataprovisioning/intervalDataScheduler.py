import datetime

from DLplatform.dataprovisioning import DataScheduler

import time


class IntervalDataScheduler(DataScheduler):
    def __init__(self, interval = 0.004, name = "IntervalDataScheduler"):
        DataScheduler.__init__(self, name = name)

        self._interval = interval

    def generateSamples(self):
        '''

        Processes next data point from training dataset, i.e. appends it to the data buffer of the worker

        Returns
        -------

        '''
        # TODO Log this ("Running", __name__) or keep track of data sample using a new variables
        #  Findings: The length of the data vector within IntervalDataScheduler keeps growing. But this does not
        #  create the large memory leak (Although it could be contributing to a smaller one)
        #  Leak looks like is happening before this method is executed

        DataScheduler.generateSamples(self)

        sent_data_length = 0
        loop_iter = 0

        while True:
            data = self.getData()
            time.sleep(self._interval)

            #if self._onDataUpdateCallBack is None:
            #    self.error("onUpdate call back function was not set")
            #    raise AttributeError("onUpdate call back function was not set")

            self.sendDataUpdate(data)

            sent_data_length += len(data)

            log_file_path = 'C:/Users/lasal/Documents/resProj/Console Logs/batchsize_logs.txt'
            log_start_time = datetime.datetime.now()
            loop_iter += 1

            if datetime.datetime.now() - log_start_time > datetime.timedelta(seconds=5):
                with open(log_file_path, 'a') as output:
                    output.write('Length of data vector at iteration ' + str(loop_iter) + ' : ' + str(sent_data_length)
                                 + '\n')