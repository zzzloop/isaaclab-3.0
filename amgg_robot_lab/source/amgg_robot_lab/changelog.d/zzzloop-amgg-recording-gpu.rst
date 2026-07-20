Fixed
^^^^^

* Fixed G1 XR demonstration recording crashes on multi-GPU workstations by
  pinning CUDA, PhysX, RTX, and CloudXR to the preferred physical GPU with
  one shared PCI-order device namespace for camera external-memory interop.

Changed
^^^^^^^

* Changed windowed recording to physical GPU 0, which owns the workstation
  presentation queue, and headless recording to physical GPU 1. Physical GPUs
  2 and 3 remain quarantined after repeated driver-level Xid failures.
