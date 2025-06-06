General state (exact) simulation
================================

.. automodule:: pytket.extensions.cutensornet.general_state
.. automodule:: pytket.extensions.cutensornet.general_state.tensor_network_state

.. autoclass:: pytket.extensions.cutensornet.general_state.tensor_network_state.GeneralState()

    .. automethod:: sample
    .. automethod:: get_amplitude
    .. automethod:: get_statevector
    .. automethod:: expectation_value
    .. automethod:: destroy

.. autoclass:: pytket.extensions.cutensornet.general_state.tensor_network_state.GeneralBraOpKet()

    .. automethod:: contract
    .. automethod:: destroy

Pytket backend
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
.. automodule:: pytket.extensions.cutensornet.backends
.. automodule:: pytket.extensions.cutensornet.backends.cutensornet_backend

    .. autoclass:: CuTensorNetStateBackend
        :members:
    
    .. autoclass:: CuTensorNetShotsBackend
        :members:

Miscellaneous
~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet.general_state.tensor_network_convert

    .. autofunction:: get_circuit_overlap
    .. autofunction:: get_operator_expectation_value
    .. autofunction:: measure_qubit_state
    .. autofunction:: measure_qubits_state
    .. autofunction:: tk_to_tensor_network
    
    .. autoclass:: TensorNetwork
        :members:
    
    .. autoclass:: ExpectationValueTensorNetwork
        :members:
    
    .. autoclass:: PauliOperatorTensorNetwork
        :members:

.. automodule:: pytket.extensions.cutensornet.general_state.utils

    .. autofunction:: circuit_statevector_postselect
    .. autofunction:: statevector_postselect
