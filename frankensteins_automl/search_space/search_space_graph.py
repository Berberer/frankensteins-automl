import logging


logger = logging.getLogger(__name__)


class SearchSpaceRestProblem(object):
    def __init__(self, unsatisfied_required_interfaces, component_mapping):
        self.unsatisfied_required_interfaces = unsatisfied_required_interfaces
        self.component_mapping = component_mapping

    @classmethod
    def from_previous_rest_problem(
        cls, rest_problem, satisfied_interface_component
    ):
        pass

    def is_satisfied(self):
        return (
            self.unsatisfied_required_interfaces is None
            or len(self.unsatisfied_required_interfaces) == 0
        )

    def get_unsatisfied_required_interfaces(self):
        pass

    def get_first_unsatisfied_required_interface(self):
        pass

    def get_component_mapping(self):
        return self.component_mapping


class SearchSpaceGraphNode(object):
    def __init__(self, predecessor, rest_problem):
        self.predecessor = predecessor
        self.successors = []
        self.rest_problem = rest_problem

    def get_predecessor(self):
        return self.predecessor

    def get_successors(self):
        return self.successors

    def set_successors(self, successors):
        self.successors = successors

    def get_rest_problem(self):
        return self.rest_problem

    def is_leaf_node(self):
        return self.rest_problem.is_satisfied()


class SearchSpaceGraphGenerator(object):
    def __init__(self, search_space, initial_component_name):
        self.search_space = search_space
        self.root_node = self.search_space.get_component_by_name(
            initial_component_name
        )

    def get_root_node(self):
        return self.root_node

    def generate_sucessors(self, node):
        if node is None:
            logger.warn("Cannot generate succesors of None")
            return None
        if node.is_leaf_node():
            logger.info("Node is a leaf node and has no successors")
            return None
        rest_problem = node.get_rest_problem()
        interface = rest_problem.get_first_unsatisfied_required_interface()[
            "name"
        ]
        components = self.search_space.get_components_providing_interface(
            interface
        )
        successors = []
        for component in components:
            logger.info(f"Generating a successor with {component.get_name()}")
            successor_rp = SearchSpaceRestProblem.from_previous_rest_problem(
                rest_problem, component
            )
            successors.append(SearchSpaceGraphNode(node, successor_rp))
        node.set_successors(successors)
        logger.info(f"{len(successors)} successors for interface {interface}")
        return successors
