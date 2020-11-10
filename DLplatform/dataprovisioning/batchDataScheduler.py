import datetime
import os

from DLplatform.dataprovisioning import DataScheduler

LOG_ON = True


class BatchDataScheduler(DataScheduler):
    def __init__(self, maxAmount, name="BatchDataScheduler"):
        DataScheduler.__init__(self, name=name)
        self.maxAmount = maxAmount

    def generateSamples(self):
        """

        Processes next data point from training dataset, i.e. appends it to the data buffer of the worker

        Returns
        -------

        """

        DataScheduler.generateSamples(self)

        # if LOG_ON:
        #     sent_data_length = 0
        #     loop_iter = 0
        #     log_file_path = '../../../../../Console Logs/batchsize_logs.txt'
        #     log_start_time = datetime.datetime.now()
        print('Starting to get data and sending to workers with maxAmount={} for PID={}'.format(self.maxAmount, self.pid))

        # while True:
        #     data = self.getData()
        #     self.sendDataUpdate(data)

        count = 0
        while count <= self.maxAmount:
            data = self.getData()
            self.sendDataUpdate(data)
            count += 1
        print('Finished sending data to workers from PID={}'.format(self.pid))

            # if datetime.datetime.now() - log_start_time > datetime.timedelta(seconds=5):
                # with open(log_file_path, 'a') as output:
                    # log_start_time = datetime.datetime.now()
                    # output.write('Process ID: ' + str(os.getpid()) + ' iteration count: ' + str(loop_iter) +
                    #              'Vector length: ' + str(sent_data_length) + '\n')
