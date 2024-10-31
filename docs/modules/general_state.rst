General state (exact) simulation
================================

.. automodule:: pytket.extensions.cutensornet.general_state

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralState()

    .. automethod:: sample
    .. automethod:: get_amplitude
    .. automethod:: get_statevector
    .. automethod:: expectation_value
    .. automethod:: destroy

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralBraOpKet()

    .. automethod:: contract
    .. automethod:: destroy

Pytket backend
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: CuTensorNetStateBackend, CuTensorNetShotsBackend
