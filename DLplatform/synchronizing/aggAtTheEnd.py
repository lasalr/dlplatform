from DLplatform.synchronizing.synchronizer import Synchronizer
from DLplatform.parameters import Parameters
from typing import List
import numpy as np

class AggregationAtTheEnd(Synchronizer):
    def __init__(self, name = "Aggregation-at-the-end"):
        Synchronizer.__init__(self, name = name)
    
    def evaluate(self, nodesDict, activeNodes: List[str]) -> (List[str], Parameters):
        if self._aggregator is None:
            self.error("No aggregator is set")
            raise AttributeError("No aggregator is set")

        # this condition is needed to call the 'evaluate' method in a standardized way across the different sync schemes
        # print('set(list(nodesDict.keys())) == set(activeNodes) evaluates to {}'.format(set(list(nodesDict.keys())) == set(activeNodes)))
        # if len(set(list(nodesDict.keys()))) > 4:
        # print('There are {} nodes in nodesDict.keys(). They are: {}'.format(len(set(list(nodesDict.keys()))), set(list(nodesDict.keys()))))
        # print('There are {} active nodes. They are: {}'.format(len(set(activeNodes)), set(activeNodes)))
        if set(list(nodesDict.keys())) == set(activeNodes):
            print('About to return activeNodes, self._aggregator(list(nodesDict.values())), {}')
            return activeNodes, self._aggregator(list(nodesDict.values())), {}
        else:
            return [], None, {}

    # def evaluate(self, nodesDict, activeNodes: List[str]) -> (List[str], Parameters):
    #     if self._aggregator is None:
    #         self.error("No aggregator is set")
    #         raise AttributeError("No aggregator is set")
    #
    #     # this condition is needed to call the 'evaluate' method in a standardized way across the different sync schemes
    #     # print('set(list(nodesDict.keys())) == set(activeNodes) evaluates to {}'.format(set(list(nodesDict.keys())) == set(activeNodes)))
    #     # if len(set(list(nodesDict.keys()))) > 4:
    #     # print('There are {} nodes in nodesDict.keys(). They are: {}'.format(len(set(list(nodesDict.keys()))), set(list(nodesDict.keys()))))
    #     # print('There are {} active nodes. They are: {}'.format(len(set(activeNodes)), set(activeNodes)))
    #     if set(list(nodesDict.keys())) == set(activeNodes):
    #         node_dict_list = list(nodesDict.values())
    #         # print('nodesDict.values():', nodesDict.values())
    #         node_vec = node_dict_list[0].get()
    #         # print('node_vec:', node_vec)
    #         node_vec = node_vec + np.random.rand(12,) * 510
    #         # print('changed node_vec:', node_vec)
    #         node_dict_list[0].set(weights=node_vec)
    #         # print('node_dict_list[0].get():', node_dict_list[0].get())
    #         print('About to return activeNodes, self._aggregator(list(nodesDict.values())), {}')
    #         return activeNodes, self._aggregator(node_dict_list), {}
    #     else:
    #         return [], None, {}

    def __str__(self):
        return "Aggregation-at-the-end synchronization"
