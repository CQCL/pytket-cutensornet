General state (exact) simulation
================================

.. automodule:: pytket.extensions.cutensornet.general_state

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralState()
    .. automethod:: __init__
    .. automethod:: get_statevector
    .. automethod:: get_amplitude
    .. automethod:: expectation_value
    .. automethod:: sample
    .. automethod:: destroy

.. autoclass:: pytket.extensions.cutensornet.general_state.GeneralBraOpKet()
    .. automethod:: __init__
    .. automethod:: contract
    .. automethod:: destroy

Pytket backend
~~~~~~~~~~~~~~

.. automodule:: pytket.extensions.cutensornet
    :members: CuTensorNetStateBackend, CuTensorNetShotsBackend
