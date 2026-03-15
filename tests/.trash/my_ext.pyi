from collections.abc import Sequence


class NodePtr:
    def __repr__(self) -> str: ...

    def __eq__(self, arg: NodePtr, /) -> bool: ...

    def __hash__(self) -> int: ...

    def get_ix(self) -> int: ...

class Circuit:
    """
    Circuits are the main class added by KLay, and require no arguments to construct.

    :code:`circuit = klay.Circuit()`
    """

    def __init__(self) -> None: ...

    def add_sdd_from_file(self, filename: str, true_lits: Sequence[int] = [], false_lits: Sequence[int] = []) -> NodePtr:
        """
        Add a sentential decision diagram (SDD) from file.

        :param filename:
        	Path to the :code:`.sdd` file on disk.
        :param true_lits:
        	List of literals that are always true and should get propagated away.
        :param false_lits:
        	List of literals that are always false and should get propagated away.
        """

    def add_d4_from_file(self, filename: str, true_lits: Sequence[int] = [], false_lits: Sequence[int] = []) -> NodePtr:
        """
        Add an NNF circuit in the D4 format from file.

        :param filename:
        	Path to the :code:`.nnf` file on disk.
        :param true_lits:
        	List of literals that are always true and should get propagated away.
        :param false_lits:
        	List of literals that are always false and should get propagated away.
        """

    def nb_nodes(self) -> int:
        """Number of nodes in the circuit."""

    def nb_root_nodes(self) -> int:
        """Number of root nodes in the circuit."""

    def true_node(self) -> NodePtr:
        """Adds a true node to the circuit, and returns a pointer to this node."""

    def false_node(self) -> NodePtr:
        """Adds a false node to the circuit, and returns a pointer to this node."""

    def literal_node(self, literal: int) -> NodePtr:
        """
        Adds a literal node to the circuit, and returns a pointer to this node.
        """

    def or_node(self, children: Sequence[NodePtr]) -> NodePtr:
        """
        Adds an :code:`or` node to the circuit, and returns a pointer to this node.
        """

    def and_node(self, children: Sequence[NodePtr]) -> NodePtr:
        """
        Adds an :code:`and` node to the circuit, and returns a pointer to this node.
        """

    def set_root(self, root: NodePtr) -> None:
        """
        Marks a node pointer as root. The order in which nodes are set as root determines the order of the output tensor.
         .. note:: Only use this when manually constructing a circuit, when loading in a NNF/SDD its root is automatically set as root.
        """

    def remove_unused_nodes(self) -> None:
        """
        Removes unused nodes from the circuit. Root nodes are always considered used.
         .. warning:: Invalidates any :code:`NodePtr` referring to an unused node (i.e., a node not connected to a root node).
        """

    def print(self) -> None:
        """Print the circuit structure to stdout."""

    def get_indices(self) -> tuple[list[list[int]], list[list[int]]]: ...

def compile_to_ganak(cnf_file: str) -> Circuit:
    """Compile a DIMACS CNF file into a klay Circuit using Ganak."""

def compile_to_ganak_debug() -> Circuit:
    """
    Compile a programmatic written CNF formula into a klay Circuit using Ganak.
    """

def compile_to_ganak_into(circuit: Circuit, cnf_file: str) -> None:
    """
    Compile a CNF file using Ganak and add the resulting nodes into an existing Circuit.
    """
