API documentation
-----------------


Full tensor network contraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: TensorNetwork, PauliOperatorTensorNetwork, ExpectationValueTensorNetwork, measure_qubits_state, tk_to_tensor_network, CuTensorNetBackend


Matrix Product State (MPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet.mps

    .. autoclass:: MPS

        .. automethod:: __init__
        .. automethod:: is_valid

    .. autoclass:: Tensor

        .. automethod:: __init__
        .. automethod:: copy