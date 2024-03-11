Full tensor network (general state) contraction
===============================================

.. automodule:: pytket.extensions.cutensornet.general_state

cuQuantum `contract` API interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.general_state.TensorNetwork

.. autoclass:: pytket.extensions.cutensornet.general_state.PauliOperatorTensorNetwork

.. autoclass:: pytket.extensions.cutensornet.general_state.ExpectationValueTensorNetwork

.. autofunction:: pytket.extensions.cutensornet.general_state.measure_qubits_state

.. autofunction:: pytket.extensions.cutensornet.general_state.tk_to_tensor_network

cuQuantum `high-level` API interface
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralState

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralOperator

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralExpectationValue

Pytket backend
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: CuTensorNetBackend
