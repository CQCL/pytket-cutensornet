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

    .. autoclass:: Tensor()

        .. automethod:: __init__
        .. automethod:: get_tensor_descriptor
        .. automethod:: copy

    .. autoclass:: DirectionMPS