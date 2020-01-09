import logging
import uuid
from abc import ABC, abstractmethod

logger = logging.getLogger(__name__)


class GraphNode(ABC):
    def __init__(self, predecessor):
        self.predecessor = predecessor
        self.successors = []
        self.node_id = str(uuid.uuid1())

    def get_node_id(self):
        return self.node_id

    def get_predecessor(self):
        return self.predecessor

    def get_successors(self):
        return self.successors

    def set_successors(self, successors):
        self.successors = successors

    @abstractmethod
    def is_leaf_node(self):
        pass


class GraphGenerator(ABC):
    @abstractmethod
    def get_root_node(self):
        pass

    def generate_successors(self, node):
        if node.is_leaf_node():
            logger.debug("Node is a leaf node and has no successors")
            return []
        if node.get_successors() != []:
            return node.get_successors()
        return self.get_node_successors(node)

    @abstractmethod
    def get_node_successors(self, node):
        pass
