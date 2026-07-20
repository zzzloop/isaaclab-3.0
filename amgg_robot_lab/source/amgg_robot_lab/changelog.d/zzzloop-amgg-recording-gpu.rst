Fixed
^^^^^

* Fixed G1 XR demonstration recording crashes on multi-GPU workstations by
  pinning CUDA, PhysX, RTX, and CloudXR to the preferred physical GPU with
  one shared PCI-order device namespace for camera external-memory interop.
* Fixed repeatable Xid 31 faults during XR camera recording by disabling
  asynchronous rendering and preventing the throttling extension from
  re-enabling it while sensor render products initialize.
* Fixed headless XR recording startup by removing the explicit Kit visualizer
  option that conflicts with deprecated ``--headless`` handling. XR still
  auto-injects the Kit visualizer required for app-update pumping.

Changed
^^^^^^^

* Changed windowed recording to physical GPU 0, which owns the workstation
  presentation queue, and headless recording to physical GPU 1. Physical GPUs
  2 and 3 remain quarantined after repeated driver-level Xid failures.
