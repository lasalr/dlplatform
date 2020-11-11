import datetime
import os
import random

from DLplatform.baseClass import baseClass
from DLplatform.learning.learner import Learner
from DLplatform.communicating import Communicator
from DLplatform.dataprovisioning import DataScheduler

import time
import pickle
from multiprocessing import Pipe, Queue
from pickle import loads
import sys


class Worker(baseClass):
    '''

    '''
    _workerCount = 0

    def __init__(self, identifier: str):
        '''

        Initialize a worker.

        Parameters
        ----------
        identifier : str

        Exception
        --------
        ValueError
            in case that identifier is not a string
        '''

        super().__init__(name="worker_" + str(identifier))

        self._learner = None
        self._communicator = None
        self._dataScheduler = None
        self._identifier = identifier
        self._dataBuffer = []

        # initializing communication with processes of communicator and dataScheduler
        self._communicatorMsgQueue = Queue()
        # dataScheduler will only write to the pipe and worker will only read,
        # so duplex is not needed; [0] is for reading, [1] for writing
        self._dataSchedulerPipe = Pipe(duplex=False)
        # for retrieval at the worker
        self._dataSchedulerRetriever = self._dataSchedulerPipe[0]
        self._workerCount += 1

    def setIdentifier(self, identifier: str):
        '''

        Set identifier of the worker.

        Parameters
        ----------
        identifier : str

        Exception
        --------
        ValueError
            in case that identifier is not a string
        '''

        if not isinstance(identifier, str):
            error_text = "The attribute identifier is of type " + str(type(identifier)) + " and not of type" + str(str)
            self.error(error_text)
            raise ValueError(error_text)

        self._identifier = identifier

    def getIdentifier(self) -> str:
        '''

        Get identifier of the worker.

        Returns
        -------
        str
        '''

        return self._identifier

    def setLearner(self, learner: Learner):
        '''

        Set learner of the worker and link communicator and worker identifier with it

        Parameters
        ----------
        learner

        Returns
        -------

        Exception
        --------
        ValueError
            in case that learner is not of type Learner
        '''
        if not isinstance(learner, Learner):
            self.error("the attribute learner is of type " + type(learner) + " and not of type" + Learner)
            raise ValueError("the attribute learner is of type " + type(learner) + " and not of type" + Learner)

        self._learner = learner
        self._learner.setIdentifier(self.getIdentifier())

    def getLearner(self) -> Learner:
        '''

        Get the learner of the worker.

        Returns
        -------
        Learner

        '''

        return self._learner

    def setCommunicator(self, comm: Communicator):
        '''

        Set the communicator of the worker.

        Parameters
        ----------
        comm

        Returns
        -------

        Exception
        --------
        ValueError
            in case that comm is not of type Communicator
        '''

        if not isinstance(comm, Communicator):
            error_text = "The attribute comm is of type " + str(type(comm)) + " and not of type" + str(Communicator)
            self.error(error_text)
            raise ValueError(error_text)

        self._communicator = comm

    def getCommunicator(self) -> Communicator:
        '''

        Get the communicator of the worker.

        Returns
        -------
        Communicator

        '''

        return self._commuicator

    def setDataScheduler(self, datascheduler: DataScheduler):
        '''

        Sets data scheduler of the worker and writes this info to the log file.

        Parameters
        ----------
        datascheduler

        Exception
        -------
        ValueError
            in case that datascheduler is not of type DataScheduler
        '''

        if not isinstance(datascheduler, DataScheduler):
            error_text = "The attribute datascheduler is of type " + str(
                type(datascheduler)) + " and not of type" + str(DataScheduler)
            self.error(error_text)
            raise ValueError(error_text)

        self._dataScheduler = datascheduler
        self.info("Set DataScheduler to " + self._dataScheduler.getName())

    def getDataScheduler(self) -> DataScheduler:
        '''

        Get data scheduler of the worker.

        Returns
        -------
        DataScheduler

        '''

        return self._dataScheduler

    def onDataUpdate(self, data: tuple):
        '''

        Defines how to process the next data point from the training dataset: Append it to the data buffer of the worker

        Parameters
        ----------
        data

        Returns
        -------

        '''
        self._dataBuffer.append(data)

    def onCommunicatorMessageReceived(self, routing_key, exchange, body):
        '''
        Processes incoming message from communicator, i.e., in federated learning setup sent from coordinator:
        In case a new model arrives, this info is logged and the previous model is replaced by that averaged model.
        In case a balancing request is sent, this info is logged and a method is called
        ('answerParameterRequest') that handles the answering of such a request
        The message might be either
        - initial model as answer to registration
        - averaged model as answer to violation or balancing process
        - averaged model together with reference model if there was a full update while balancing
        - request to send parameters

        Parameters
        ----------
        default parameters from RabbitMQ for callback; body contains the message itself

        Returns
        -------
        None
        '''

        self.info('Got message in the worker queue')
        # print('onCommunicatorMessageReceived() is being called')
        # print('routing_key =', routing_key)
        if 'newModel' in routing_key:
            # print('newModel is in routing_key')
            body_size = sys.getsizeof(body)
            self._communicator.learningLogger.logSendModelMessage(exchange, routing_key, body_size, 'receive',
                                                                  self.getIdentifier())
            self.info("The learner received initial setup or averaged model, with or without reference model")
            message = pickle.loads(body)
            param = message['param']
            flags = message['flags']
            # print('setModel() is about to be called')
            self._learner.setModel(param, flags)
        if 'request' in routing_key:
            body_size = 0
            self._communicator.learningLogger.logBalancingRequestMessage(exchange, routing_key, body_size, 'receive',
                                                                         self.getIdentifier())
            self.info("Coordinator asks for parameters to balance violation")
            self._learner.answerParameterRequest()
        if 'exit' in routing_key:
            body_size = 0
            self.info("Coordinator stops the execution")
            self._learner.stopExecution()

    def checkInterProcessCommunication(self):
        '''
        Checks pipe and queue for new incoming messages and acts in case if a message has arrived.
        Can be message from communicator, from queue; or message from dataScheduler, from pipe

        Exceptions
        ----------
        ValueError
            in case that the received message doesn't fit with the expected type
        '''
        # print('checkInterProcessCommunication is being called')
        if not self._communicatorMsgQueue.empty():
            # message from communicator is not pickled, since it is already simple
            # objects, that can be passed through external means of messaging
            recvObj = self._communicatorMsgQueue.get()

            if not isinstance(recvObj, tuple):
                raise ValueError("worker received recvObj that is not a tuple")
            elif not len(recvObj) == 3:
                raise ValueError("worker received recvObj which has length different from 3")

            # we know exactly the structure of the message from communicator
            routing_key, exchange, body = recvObj
            self.onCommunicatorMessageReceived(routing_key, exchange, body)

        if self._dataSchedulerRetriever.poll():
            # receive next training example
            recvObj = self._dataSchedulerRetriever.recv()
            value = loads(recvObj)
            self._dataBuffer.append(value)

    def _setConnectionsToComponents(self):
        '''

        distributes the transmitters and receiver connections over the different processes such that an inter process
        communication can take place.

        Exceptions
        ----------
        AttributeError
            in case if either no dataScheduler, no communicator or no learner is set

        '''

        if self._dataScheduler is None:
            self.error("DataScheduler not set!")
            raise AttributeError("DataScheduler not set!")

        if self._communicator is None:
            self.error("Communicator not set!")
            raise AttributeError("Communicator not set!")

        if self._learner is None:
            self.error("Learner not set!")
            raise AttributeError("Learner not set!")

        self._communicator.setConnection(consumerConnection=self._communicatorMsgQueue)
        self._dataScheduler.setConnection(workerConnection=self._dataSchedulerPipe[1])

    def run(self):
        '''
        Configures and starts data scheduler and communicator of the worker.
        Requests initial model from the coordinator.
        Continuously transfers training data points from data buffer to learner.
        The actual operation logic of the worker

        Returns
        -------

        Exception
        ---------
        AttributeError
            In case that at least one of the necessary modules DataScheduler, Communicator or Learner is not set
            In case that at least on of the necessary modules DataScheduler, communicator or Learner is not set or in
            case that the connection from dataScheduler or to the communicator aren't set.
        '''

        if self._dataScheduler is None:
            self.error("DataScheduler not set!")
            raise AttributeError("DataScheduler not set!")

        if self._communicator is None:
            self.error("Communicator not set!")
            raise AttributeError("Communicator not set!")

        if self._learner is None:
            self.error("Learner not set!")
            raise AttributeError("Learner not set!")

        # learner needs an communicator object to publish detailed messages, that worker cannot do
        # e.g., only learner knows what exact class of parameters it has, worker knows only virtual class
        self._learner.setCommunicator(self._communicator)

        # dataScheduler is for individual setup of giving data to the worker
        # it is running in its own process since the data is constantly generated, independent from the learner
        self._dataScheduler.daemon = True

        # communicator runs in thread to consume the queue of the worker
        self._communicator.initiate(exchange=self._communicator._exchangeNodes,
                                    topics=["#." + self.getIdentifier() + ".#", "#." + self.getIdentifier()])
        self._communicator.daemon = True

        self._setConnectionsToComponents()

        self._dataScheduler.start()
        self._communicator.start()

        if (self._communicatorMsgQueue == None) or (self._dataSchedulerRetriever == None):
            raise AttributeError("either communicator connection or dataScheduler connection was not set properly at "
                                 "the worker!")

        # initializing of consumer of the communicator takes time...
        print('Worker {} is sleeping to allow time for communicator to start...'.format(self._identifier))
        time.sleep(5)
        # only now we should request for initial model - or we will not be able to receive the answer
        self._learner.requestInitialModel()

        # TODO check what happens in this loop once sys.exit() is run in learner.py.
        #  Does the whole loop end and  processes all die?
        # while not self._learner._stop:
        while self._learner.isAlive():
            self.checkInterProcessCommunication()
            if len(self._dataBuffer) > 0:
                if self._learner.canObtainData():
                    self._learner.obtainData(self._dataBuffer[0])
                    del(self._dataBuffer[0])
                # else:
                #     # Sleep worker for a bit since len(self._dataBuffer) <= 0'.format(self._identifier))
                #     time.sleep(self._workerCount/10)  # TODO see if this is required

        print('Local training complete for node with identifier, self._identifier =', self._identifier)
        self._dataScheduler.terminate()
        self._dataScheduler.join()
        self._communicator.terminate()
        self._communicator.join()
        print('worker ', self._identifier, ' shut down.')
