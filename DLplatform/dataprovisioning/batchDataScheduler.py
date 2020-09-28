import datetime
import os
import tracemalloc

from DLplatform.dataprovisioning import DataScheduler

MEM_TRACE = False
LOG_ON = True


class BatchDataScheduler(DataScheduler):
    def __init__(self, name="BatchDataScheduler"):
        DataScheduler.__init__(self, name=name)

    def generateSamples(self):
        """

        Processes next data point from training dataset, i.e. appends it to the data buffer of the worker

        Returns
        -------

        """

        if MEM_TRACE:
            tracemalloc.start(100)
            trace_start_time = datetime.datetime.now()

        DataScheduler.generateSamples(self)

        if LOG_ON:
            sent_data_length = 0
            loop_iter = 0
            log_file_path = 'C:/Users/lasal/Documents/resProj/Console Logs/batchsize_logs.txt'
            log_start_time = datetime.datetime.now()



        if MEM_TRACE:
            if datetime.datetime.now() - trace_start_time > datetime.timedelta(seconds=10):
                snapshot = tracemalloc.take_snapshot()
                top_stats = snapshot.statistics('lineno')
                print("Process ID:", str(os.getpid()), "[ Top 10 ] - while-true in generateSamples() in "
                                                       "BatchDataScheduler")
                for stat in top_stats[:10]:
                    print(stat)

        while True:
            data = self.getData()
            self.sendDataUpdate(data)
            sent_data_length += len(data)

            loop_iter += 1

            if datetime.datetime.now() - log_start_time > datetime.timedelta(seconds=5):
                with open(log_file_path, 'a') as output:
                    log_start_time = datetime.datetime.now()
                    output.write('Process ID: ' + str(os.getpid()) + ' iteration count: ' + str(loop_iter) +
                                 'Vector length: ' + str(sent_data_length) + '\n')
