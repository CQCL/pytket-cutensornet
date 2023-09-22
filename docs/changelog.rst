Changelog
~~~~~~~~~

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
