Fixed
^^^^^

* Fixed G1 XR demonstration recording crashes on multi-GPU workstations by
  pinning CUDA, PhysX, RTX, and CloudXR to the preferred physical GPU with
  one shared PCI-order device namespace for camera external-memory interop.

Changed
^^^^^^^

* Changed the default recording device to physical GPU 1 and quarantined
  physical GPUs 2 and 3 after repeated driver-level Xid failures.
