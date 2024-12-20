"""
External routing software using Cirq greedy routing
"""

import networkx as nx

import cirq.contrib.routing.greedy
import qiskit, qiskit.transpiler.exceptions
import pytket, pytket.qasm

import warnings
import logging
import time
logging.basicConfig(level='CRITICAL')
warnings.simplefilter("ignore", category=DeprecationWarning)


def cirq_routing(circuit, device):
    """
    Solves the qubit routing problem using Cirq greedy routing
    :param circuit: the input logical circuit to route
    :param device: the device we are trying to compile to
    :return: swap circuit, like an actual circuit but with swap operations inserted with logical nomenclature
    """

    device_graph = nx.Graph()
    for edge in device.edges:
        device_graph.add_edges_from([(cirq.LineQubit(edge[0]), cirq.LineQubit(edge[1]))])
    swap_network = cirq.contrib.routing.greedy.route_circuit_greedily(circuit.cirq, device_graph)
    return len(swap_network.circuit.moments)


def qiskit_routing(circuit, device):
    """
    Routing using qiskit
    :param circuit: Circuit object, Either qasm filepath or Cirq's circuit object
    :param device: Device object, Defines topology for hardware specific routing
    :returns depth: int, Circuit depth according to tket routing strategies
    """

    q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
    coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
    routing = {'basic': 0, 'stochastic': 0, 'sabre': 0}
    #routing = {'stochastic': 0}
    for rt in routing.keys():
        try:
            tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=rt,
                                                   optimization_level=0)
            routing.update({rt: tr_circuit.depth()})
        except qiskit.transpiler.exceptions.TranspilerError:
            routing.update({rt: 999999999})
        #print(qiskit_routing_info(circuit,device))
    return routing


def qiskit_routing_stochastic(circuit, device):
    q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
    coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
    routing = 'stochastic'
    tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=routing,
                                                   optimization_level=0)
    return tr_circuit.depth()

def qiskit_routing_info(circuit, device):
    q_circuit = qiskit.QuantumCircuit.from_qasm_str(circuit.cirq.to_qasm(header=''))
    coupling_map = list(map(list, device.edges)) + list(map(list, map(reversed, device.edges)))
    routing = 'stochastic'
    tr_circuit = qiskit.compiler.transpile(q_circuit, coupling_map=coupling_map, routing_method=routing,
                                                   optimization_level=0)
    
    swaps = []
    for gate, qargs, cargs in tr_circuit.data:
        if gate.name == 'swap':
            qubit1 = qargs[0]._index
            qubit2 = qargs[1]._index
            swaps.append((qubit1, qubit2))
    
    # Initialize the list of edges where swaps are made
    edges_involved_in_swaps = [False] * len(device.edges)
    
    # Check which edges were involved in swaps
    for swap in swaps:
        for idx, edge in enumerate(device.edges):
            if (swap[0], swap[1]) == edge or (swap[1], swap[0]) == edge:
                edges_involved_in_swaps[idx] = True

    swap_indices = [idx for idx, involved in enumerate(edges_involved_in_swaps) if involved]

    return swap_indices,edges_involved_in_swaps


def tket_routing(circuit, device):
    """
    Routing using pytket
    :param circuit: Circuit object, Either qasm filepath or Cirq's circuit object
    :param device: Device object, Defines topology for hardware specific routing
    :returns depth: int, Circuit depth according to tket routing strategies
    """

    tket_circuit = pytket.qasm.circuit_from_qasm_str(circuit.cirq.to_qasm(header=''))
    transformer = pytket.transform.Transform
    architecture = pytket.routing.Architecture(
        list(map(list, device.edges)) + list(map(list, map(reversed, device.edges))))
    routed_circuit = pytket.routing.route(circuit=tket_circuit, architecture=architecture)
    # noinspection PyArgumentList
    transformer.DecomposeBRIDGE().apply(routed_circuit)
    # noinspection PyArgumentList
    transformer.RemoveRedundancies().apply(routed_circuit)
    return routed_circuit.depth()
