API documentation
-----------------


Full tensor network contraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: TensorNetwork, PauliOperatorTensorNetwork, ExpectationValueTensorNetwork, measure_qubits_state, tk_to_tensor_network, CuTensorNetBackend


Matrix Product State (MPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet.mps

    .. autoclass:: MPS()

        .. automethod:: __init__
        .. automethod:: init_cutensornet
        .. automethod:: apply_gate
        .. automethod:: vdot
        .. automethod:: canonicalise
        .. automethod:: is_valid
        .. automethod:: __len__
        .. automethod:: copy

    .. autoclass:: MPSxGate

    .. autoclass:: MPSxMPO

    .. autoenum:: ContractionAlg()
        :members:

    .. automethod:: pytket.extensions.cutensornet.mps.simulate

    .. automethod:: pytket.extensions.cutensornet.mps.get_amplitude

    .. automethod:: pytket.extensions.cutensornet.mps.prepare_circuit

    .. autoclass:: Tensor()

        .. automethod:: __init__
        .. automethod:: get_tensor_descriptor
        .. automethod:: copy

    .. autoenum:: DirectionMPS()
        :members:
