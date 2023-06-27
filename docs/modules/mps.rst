Matrix Product State (MPS)
==========================

.. automodule:: pytket.extensions.cutensornet.mps

Simulation
----------

    .. autoenum:: ContractionAlg()
        :members:

    .. automethod:: pytket.extensions.cutensornet.mps.simulate

    .. automethod:: pytket.extensions.cutensornet.mps.get_amplitude

Classes
-------

    .. autoclass:: MPS()

        .. automethod:: __init__
        .. automethod:: init_cutensornet
        .. automethod:: apply_gate
        .. automethod:: vdot
        .. automethod:: canonicalise
        .. automethod:: is_valid
        .. automethod:: copy
        .. automethod:: __len__

    .. autoclass:: MPSxGate()
        :show-inheritance:

        .. automethod:: __init__

    .. autoclass:: MPSxMPO()
        :show-inheritance:

        .. automethod:: __init__

    .. autoclass:: Tensor()

        .. automethod:: __init__
        .. automethod:: get_tensor_descriptor
        .. automethod:: copy

    .. autoenum:: DirectionMPS()
        :members:

Miscellaneous
-------------

    .. automethod:: pytket.extensions.cutensornet.mps.prepare_circuit