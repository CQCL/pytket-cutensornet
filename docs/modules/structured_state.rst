Structured state evolution
==========================

.. automodule:: pytket.extensions.cutensornet.structured_state

Library handle
~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.CuTensorNetHandle

    .. automethod:: destroy


Simulation
~~~~~~~~~~

.. autofunction:: pytket.extensions.cutensornet.structured_state.simulate

.. autoenum:: pytket.extensions.cutensornet.structured_state.SimulationAlgorithm()
    :members:

.. autoclass:: pytket.extensions.cutensornet.structured_state.Config()

    .. automethod:: __init__


Classes
~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.structured_state.StructuredState()

    .. automethod:: is_valid
    .. automethod:: apply_gate
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

.. autoclass:: pytket.extensions.cutensornet.structured_state.TTNxGate()

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.structured_state.MPSxGate()

    .. automethod:: __init__
    .. automethod:: add_qubit

.. autoclass:: pytket.extensions.cutensornet.structured_state.MPSxMPO()

    .. automethod:: __init__


Miscellaneous
~~~~~~~~~~~~~

.. autofunction:: pytket.extensions.cutensornet.structured_state.prepare_circuit_mps
