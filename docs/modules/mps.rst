Matrix Product State (MPS)
==========================

.. automodule:: pytket.extensions.cutensornet.mps


Simulation
~~~~~~~~~~

.. autoenum:: pytket.extensions.cutensornet.mps.ContractionAlg()
    :members:

.. autofunction:: pytket.extensions.cutensornet.mps.simulate

.. autofunction:: pytket.extensions.cutensornet.mps.get_amplitude


Classes
~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.mps.MPS()

    .. automethod:: __init__
    .. automethod:: set_libhandle
    .. automethod:: apply_gate
    .. automethod:: vdot
    .. automethod:: canonicalise
    .. automethod:: get_virtual_bonds
    .. automethod:: get_virtual_dimensions
    .. automethod:: get_physical_bond
    .. automethod:: get_physical_dimension
    .. automethod:: get_device_id
    .. automethod:: is_valid
    .. automethod:: copy
    .. automethod:: __len__

.. autoclass:: pytket.extensions.cutensornet.mps.MPSxGate()
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.mps.MPSxMPO()
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.mps.CuTensorNetHandle

.. autoclass:: pytket.extensions.cutensornet.mps.Tensor()

    .. automethod:: __init__
    .. automethod:: get_bond_dimension
    .. automethod:: get_tensor_descriptor
    .. automethod:: copy


Miscellaneous
~~~~~~~~~~~~~

.. autoenum:: pytket.extensions.cutensornet.mps.DirectionMPS()
    :members:

.. autofunction:: pytket.extensions.cutensornet.mps.prepare_circuit
