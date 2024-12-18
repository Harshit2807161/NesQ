import nesq


def test_generator_full_layer():
    cirq_circuit = nesq.environment.circuits.circuit_generated_full_layer(6)
    circuit = nesq.environment.circuits.CircuitRepDQN(cirq_circuit)
    targets = []
    for target_list in circuit.circuit:
        assert type(targets) is list, "The Circuit DQN should be a list of lists"
        assert len(target_list) == 1, "Expecting one operation in single layer circuit"
        targets.append(target_list[0])
    assert sorted(targets) == list(range(6)), "All operations not completed"
