Changelog
~~~~~~~~~

0.5.1 (December 2023)
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
