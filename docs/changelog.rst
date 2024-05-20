Changelog
~~~~~~~~~

Unreleased
----------

* New feature: MPS algorithms ``MPSxGate`` and ``MPSxMPO`` now support simulation of two-qubit gates acting on non-adjacent qubits.
* New feature: ``add_qubit`` to add fresh qubits at specified positions in an ``MPS``.
* New feature: added an option to ``measure`` to toggle destructive measurement on/off. Currently only supported for ``MPS``.
* New feature: ``apply_unitary`` both for ``MPS`` and ``TTN`` to apply an arbitrary unitary matrix, rather than a ``pytket.Command``.
* New feature: ``apply_qubit_relabelling`` both for ``MPS`` and ``TTN`` to change the name of their qubits. This is now used within ``simulate`` to take into account the action of implicit SWAPs in pytket circuits (no additional SWAP gates are applied).

0.6.1 (April 2024)
------------------

* When using ``simulate`` with ``TTNxGate`` algorithm, the initial partition is obtained using NetworkX instead of KaHyPar by default. This makes setup easier and means that ``TTNxGate`` can now be used when installing from PyPI. KaHyPar can still be used if ``use_kahypar`` from ``Config`` is set to True.
* Updated pytket version requirement to 1.27.

0.6.0 (April 2024)
------------------

* **New feature**: Tree Tensor Network (TTN) simulator, supporting both fixed ``chi`` and ``truncation_fidelity``. Calculation of single amplitudes is supported by ``get_amplitude`` and inner products by ``vdot``. Measurement and postselection are not yet supported.
* **New API**: both ``MPS`` and ``TTN`` share a common interface: ``StructuredState``. Import paths have changed, multiple classes have been renamed: ``ConfigMPS`` is now ``Config``, ``ContractionAlg`` is now ``SimulationAlgorithm``. Documentation has been updated accordingly.

* Canonicalisation of MPS is now always applied before a two-qubit gate. We found that this tends to reduce runtime due to canonicalisation decreasing virtual bond dimension.
* Two-qubit gates are now decomposed (SVD) before applying them to remove null singular values (e.g. in ``XXPhase`` gates).
* Fixed a bug on copying an ``MPS`` if ``truncation_fidelity`` was set.
* Fixed a bug on ``CuTensorNetHandle`` that would prevent it from working when the device set was different from the default one (``dev=0``) and when using ``cuTensorNet>=2.3.0``.
* Fixed a bug on ``TensorNetwork`` due to unsupported ``Create`` operation.
* Updated pytket version requirement to 1.26.

0.5.4 (January 2024)
--------------------

* Updated pytket version requirement to 1.24.
* Python 3.12 support added, 3.9 dropped.

0.5.3 (January 2024)
--------------------

* Updated pytket version requirement to 1.23.

0.5.2 (December 2023)
---------------------

* ``MPS`` simulation with fixed ``truncation_fidelity`` now uses the corresponding truncation primitive from cuQuantum (v23.10).
* Updated pytket version requirement to 1.22.

0.4.0 (October 2023)
--------------------

* API Update. Configuration of ``MPS`` simulation parameters is now done via ``ConfigMPS``.
* Added a ``value_of_zero`` parameter to ``ConfigMPS`` for the user to indicate the threshold below which numbers are so small that can be interpreted as zero.
* Added a logger to MPS methods. Use it by setting ``loglevel`` in ``ConfigMPS``.
* Improved performance of contraction across ``MPS`` methods by hardcoding the contraction paths.
* Fixed a bug that caused more MPS canonicalisation than strictly required.
* Fixed a bug where ``simulate`` would not apply the last batch of gates when using ``MPSxMPO``.

0.3.0 (September 2023)
----------------------

* Added MPS sampling feature.
* Refactored MPS module for better maintainability and extendability.
* ``Tensor`` class removed from the API since it is no longer necessary.

0.2.1 (August 2023)
-------------------

* Improved backend gate set to allow for more gate types.
* Fixed a bug in ``apply_gate`` of MPS algorithms that would cause internal dimensions to be tracked wrongly in certain edge cases, causing a crash.

0.2.0 (July 2023)
-----------------

* Added post selection capability for expectation value tensor networks.
* Added MPS simulation approaches, supporting two contraction algorithms (gate-by-gate and DMRG-like). Supports exact simulation, as well as approximate simulation with either fixed virtual bond dimension or target gate fidelity.

0.1.0 (June 2023)
-----------------

* Initial implementation of the converter and backend modules for use on a single GPU.
