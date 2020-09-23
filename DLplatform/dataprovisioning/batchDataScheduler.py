import datetime
import tracemalloc
from DLplatform.dataprovisioning import DataScheduler
import time
MEM_TRACE = False
class BatchDataScheduler(DataScheduler):
    def __init__(self, name="BatchDataScheduler"):
        DataScheduler.__init__(self, name=name)
    def generateSamples(self):
        '''

        Processes next data point from training dataset, i.e. appends it to the data buffer of the worker

        Returns
        -------





        '''
        if MEM_TRACE:
            tracemalloc.start(100)
            trace_start_time = datetime.datetime.now()

        DataScheduler.generateSamples(self)

            if datetime.datetime.now() - log_start_time > datetime.timedelta(seconds=5):
                with open(log_file_path, 'a') as output:
                    trace_start_time = datetime.datetime.now()
                    output.write('Length of data vector at iteration ' + str(loop_iter) + ' : ' + str(sent_data_length)
                                 + '\n')

            if MEM_TRACE:
                if datetime.datetime.now() - trace_start_time > datetime.timedelta(seconds=10):
                    snapshot = tracemalloc.take_snapshot()
                    top_stats = snapshot.statistics('lineno')
                    print("Process ID:", str(os.getpid()), "[ Top 10 ] - while-true in generateSamples() in "
                                                           "BatchDataScheduler")
                    for stat in top_stats[:10]:
                        print(stat)
            data = self.getData()
            self.sendDataUpdate(data)
            sent_data_length += len(data)

            log_file_path = 'C:/Users/lasal/Documents/resProj/Console Logs/batchsize_logs.txt'
            log_start_time = datetime.datetime.now()
            loop_iter += 1

        sent_data_length = 0
        loop_iter = 0

        while True:
