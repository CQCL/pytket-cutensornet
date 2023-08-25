Matrix Product State (MPS)
==========================

.. automodule:: pytket.extensions.cutensornet.mps


Simulation
~~~~~~~~~~

.. autoenum:: pytket.extensions.cutensornet.mps.ContractionAlg()
    :members:

.. autofunction:: pytket.extensions.cutensornet.mps.simulate


Classes
~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.mps.MPS()

    .. automethod:: __init__
    .. automethod:: apply_gate
    .. automethod:: vdot
    .. automethod:: canonicalise
    .. automethod:: sample
    .. automethod:: measure
    .. automethod:: postselect
    .. automethod:: expectation_value
    .. automethod:: get_statevector
    .. automethod:: get_amplitude
    .. automethod:: get_qubits
    .. automethod:: get_virtual_dimensions
    .. automethod:: get_physical_dimension
    .. automethod:: get_device_id
    .. automethod:: is_valid
    .. automethod:: update_libhandle
    .. automethod:: copy
    .. automethod:: __len__

.. autoclass:: pytket.extensions.cutensornet.mps.MPSxGate()
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.mps.MPSxMPO()
    :show-inheritance:

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.mps.CuTensorNetHandle


Miscellaneous
~~~~~~~~~~~~~

.. autoenum:: pytket.extensions.cutensornet.mps.DirectionMPS()
    :members:

.. autofunction:: pytket.extensions.cutensornet.mps.prepare_circuit
