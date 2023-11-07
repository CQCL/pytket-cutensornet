TN state evolution
==================

.. automodule:: pytket.extensions.cutensornet.tnstate


Simulation
~~~~~~~~~~

.. autofunction:: pytket.extensions.cutensornet.tnstate.simulate

.. autoenum:: pytket.extensions.cutensornet.tnstate.SimulationAlgorithm()
    :members:

.. autoclass:: pytket.extensions.cutensornet.tnstate.Config()

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.tnstate.CuTensorNetHandle


Classes
~~~~~~~

.. autoclass:: pytket.extensions.cutensornet.tnstate.TNState()

    .. automethod:: __init__
    .. automethod:: is_valid
    .. automethod:: apply_gate
    .. automethod:: apply_scalar
    .. automethod:: vdot
    .. automethod:: sample
    .. automethod:: measure
    .. automethod:: postselect
    .. automethod:: expectation_value
    .. automethod:: get_fidelity
    .. automethod:: get_statevector
    .. automethod:: get_amplitude
    .. automethod:: get_qubits
    .. automethod:: get_byte_size
    .. automethod:: get_device_id
    .. automethod:: update_libhandle
    .. automethod:: copy

.. autoclass:: pytket.extensions.cutensornet.tnstate.TTNxGate()

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.tnstate.MPSxGate()

    .. automethod:: __init__

.. autoclass:: pytket.extensions.cutensornet.tnstate.MPSxMPO()

    .. automethod:: __init__


Miscellaneous
~~~~~~~~~~~~~

.. autofunction:: pytket.extensions.cutensornet.tnstate.prepare_circuit_mps
