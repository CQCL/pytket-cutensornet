General state (exact) simulation
================================

.. automodule:: pytket.extensions.cutensornet.general_state

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralState()

    .. automethod:: __init__
    .. automethod:: get_statevector
    .. automethod:: expectation_value
    .. automethod:: sample
    .. automethod:: destroy

cuQuantum `contract` API interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.general_state.TensorNetwork

.. autoclass:: pytket.extensions.cutensornet.general_state.PauliOperatorTensorNetwork

.. autoclass:: pytket.extensions.cutensornet.general_state.ExpectationValueTensorNetwork

.. autofunction:: pytket.extensions.cutensornet.general_state.tk_to_tensor_network

.. autofunction:: pytket.extensions.cutensornet.general_state.measure_qubits_state

.. autofunction:: pytket.extensions.cutensornet.general_state.get_operator_expectation_value

.. autofunction:: pytket.extensions.cutensornet.general_state.get_circuit_overlap


Pytket backend
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: CuTensorNetStateBackend, CuTensorNetShotsBackend
