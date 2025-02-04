.. currentmodule:: pytket.extensions.cutensornet

Changelog
~~~~~~~~~

0.10.2 (February 2025)
----------------------

* Small performance improvements on non-adjacent two-qubit gate application.
* Contrained pytket version to 1.x.
* Updated pytket version requirement to 1.39.

0.10.1 (December 2024)
----------------------

* Now supporting ``ClExpr`` operations (the new version of tket's ``ClassicalExpBox``).
* Updated pytket version requirement to 1.38.0.

0.10.0 (October 2024)
---------------------

* New API: ``GeneralBraOpKet`` for exact calculation of arbitrary ``<bra|op|ket>`` values. Can be used to calculate inner products, expectation values and arbitrary matrix elements.
* New feature: both ``GeneralState`` and ``GeneralBraOpKet`` admit circuits with parameterised gates.
* New feature: ``GeneralState`` has a new method ``get_amplitude`` to obtain the amplitude of computational basis states.
* New feature: ``GeneralState`` and ``CuTensorNetShotsBackend`` now support RNG seeds for sampling.
* Docs: three tutorials added to the documentation.
* Deprecated ``TensorNetwork`` object. It is still available for the sake of backwards compatibility, but it has been removed from doc pages.

0.9.0 (October 2024)
---------------------

* Updated pytket version requirement to 1.33.

0.8.0 (September 2024)
----------------------

* API breaking changes
    * Removed ``use_kahypar`` option from ``Config``. It can still be set via the ``simulate`` option ``compilation_params``.

* New feature: ``simulate`` now accepts pytket circuits with ``Measure``, ``Reset``, ``Conditional``, ``ClassicalExpBox`` and more classical operations. You can now retrieve classical bit values using ``get_bits``.
* When calling ``simulate``, the gates on the circuit are no longer sorted by default. Use ``compilation_params["sort_gates"] = True`` to recover this behaviour, which is now deprecated.
* ``StructuredState`` now supports simulation of single qubit circuits.
* Some bugfixes on ``MPSxMPO`` relating to measurement and relabelling qubits. The bug was caused due to these functions not guaranteeing the MPO was applied before their action.
* Documentation fixes:
    * ``apply_qubit_relabelling`` now appears in the documentation.
    * ``add_qubit`` removed from documentation of ``MPSxMPO``, since it is not currently supported.

0.7.1 (July 2024)
-----------------

* New official `documentation site <https://docs.quantinuum.com/tket/extensions/pytket-cutensornet/>`_.
* Backend methods can now be given a ``scratch_fraction`` argument to configure the amount of GPU memory allocated to cuTensorNet contraction. Users can also configure the values of the ``StateAttribute`` and ``SamplerAttribute`` from cuTensornet via the backend interface.
* Fixed a bug causing the logger to fail displaying device properties.

0.7.0 (July 2024)
-----------------

* API breaking changes
    * Renamed ``CuTensorNetBackend`` to ``CuTensorNetStateBackend``.
    * Moved ``get_operator_expectation_value`` and ``get_circuit_overlap`` from ``backends`` submodule to ``general_state`` submodule.
    * **Warning** ``TensorNetwork`` object will soon be deprecated in favour of the new ``GeneralState``.

* New API: ``GeneralState`` for exact simulation of circuits via contraction-path optimisation. Currently supports ``get_statevector()``, ``expectation_value()`` and ``sample()``.
* New feature: ``CuTensorNetShotsBackend`` for simulation of circuit shots.
* New feature: MPS algorithms ``MPSxGate`` and ``MPSxMPO`` now support simulation of two-qubit gates acting on non-adjacent qubits.
* New feature: ``add_qubit`` to add fresh qubits at specified positions in an ``MPS``.
* New feature: added an option to ``measure`` to toggle destructive measurement on/off. Currently only supported for ``MPS``.
* New feature: a seed can now be provided to ``Config`` objects, providing reproducibility across ``StructuredState`` simulations.
* New feature: ``apply_unitary`` both for ``MPS`` and ``TTN`` to apply an arbitrary unitary matrix, rather than a ``pytket.Command``.
* New feature: ``apply_qubit_relabelling`` both for ``MPS`` and ``TTN`` to change the name of their qubits. This is now used within ``simulate`` to take into account the action of implicit SWAPs in pytket circuits (no additional SWAP gates are applied).
* Updated pytket version requirement to 1.30.

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
