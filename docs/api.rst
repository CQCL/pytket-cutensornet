API documentation
-----------------


Full tensor network contraction
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.TensorNetwork
    :members:

.. automethod:: tk_to_tensor_network

.. automethod:: measure_qubits_state

.. autoclass:: pytket.extensions.cutensornet.PauliOperatorTensorNetwork
    :members:

.. autoclass:: pytket.extensions.cutensornet.ExpectationValueTensorNetwork
    :members:

.. autoclass:: pytket.extensions.cutensornet.CuTensorNetBackend


Matrix Product State (MPS)
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.mps.MPS
    :members:

.. autoclass:: DirectionMPS

.. autoclass:: ContractionAlg

.. autoclass:: MPSxGate

.. autoclass:: MPSxMPO

.. automethod:: simulate

.. automethod:: get_amplitude