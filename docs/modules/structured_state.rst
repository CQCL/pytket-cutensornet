Structured state evolution
==========================

.. automodule:: pytket.extensions.cutensornet.structured_state

Library handle
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet.general

    .. autofunction:: set_logger
    
    .. autoclass:: CuTensorNetHandle
        :members:


Simulation
~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet.structured_state.simulation
.. autofunction:: pytket.extensions.cutensornet.structured_state.simulation.simulate

.. autoenum:: pytket.extensions.cutensornet.structured_state.SimulationAlgorithm()
    :members:

.. automodule:: pytket.extensions.cutensornet.structured_state.general
.. autoclass:: pytket.extensions.cutensornet.structured_state.general.Config()

    .. automethod:: __init__
    .. automethod:: copy


Classes
~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.structured_state.general.StructuredState()

    .. automethod:: is_valid
    .. automethod:: apply_gate
    .. automethod:: apply_cnx
    .. automethod:: apply_pauli_gadget
    .. automethod:: apply_unitary
    .. automethod:: apply_scalar
    .. automethod:: apply_qubit_relabelling
    .. automethod:: vdot
    .. automethod:: sample
    .. automethod:: measure
    .. automethod:: postselect
    .. automethod:: expectation_value
    .. automethod:: get_fidelity
    .. automethod:: get_statevector
    .. automethod:: get_amplitude
    .. automethod:: get_bits
    .. automethod:: get_qubits
    .. automethod:: get_byte_size
    .. automethod:: get_device_id
    .. automethod:: update_libhandle
    .. automethod:: copy

.. automodule:: pytket.extensions.cutensornet.structured_state.mps
.. autoclass:: pytket.extensions.cutensornet.structured_state.mps.MPS
    :members:

.. autoclass:: pytket.extensions.cutensornet.structured_state.DirMPS
    :members:

.. automodule:: pytket.extensions.cutensornet.structured_state.ttn_gate
.. autoclass:: pytket.extensions.cutensornet.structured_state.TTNxGate()

    .. automethod:: __init__

.. automodule:: pytket.extensions.cutensornet.structured_state.mps_gate
.. autoclass:: pytket.extensions.cutensornet.structured_state.mps_gate.MPSxGate()

    .. automethod:: __init__
    .. automethod:: add_qubit
    .. automethod:: get_entanglement_entropy
    .. automethod:: apply_cnx
    .. automethod:: apply_pauli_gadget
    .. automethod:: measure_pauli_string

.. automodule:: pytket.extensions.cutensornet.structured_state.mps_mpo
.. autoclass:: pytket.extensions.cutensornet.structured_state.mps_mpo.MPSxMPO()

    .. automethod:: __init__
    .. automethod:: add_qubit
    .. automethod:: apply_qubit_relabelling
    .. automethod:: get_physical_dimension
    .. automethod:: update_libhandle

.. automodule:: pytket.extensions.cutensornet.structured_state.ttn
.. autoclass:: pytket.extensions.cutensornet.structured_state.ttn.TTN
    :members:

.. autoclass:: pytket.extensions.cutensornet.structured_state.ttn.DirTTN
    :members:

.. autoclass:: pytket.extensions.cutensornet.structured_state.ttn.TreeNode
    
    .. automethod:: copy

Miscellaneous
~~~~~~~~~~~~~

.. autofunction:: pytket.extensions.cutensornet.structured_state.simulation.prepare_circuit_mps
.. automodule:: pytket.extensions.cutensornet.structured_state.classical
.. autofunction:: pytket.extensions.cutensornet.structured_state.classical.apply_classical_command
.. autofunction:: pytket.extensions.cutensornet.structured_state.classical.evaluate_clexpr
.. autofunction:: pytket.extensions.cutensornet.structured_state.classical.from_little_endian
.. autoexception:: pytket.extensions.cutensornet.structured_state.LowFidelityException
.. automodule:: pytket.extensions.cutensornet._metadata
