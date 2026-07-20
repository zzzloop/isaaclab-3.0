Fixed
^^^^^

* Fixed G1 XR demonstration recording crashes on multi-GPU workstations by
  pinning CUDA, PhysX, RTX, and CloudXR to the preferred physical GPU with
  one shared PCI-order device namespace for camera external-memory interop.
* Fixed repeatable Xid 31 faults during XR camera recording by synchronizing
  Replicator sensor capture and preventing the throttling extension from
  toggling asynchronous rendering while render products initialize. Headless
  capture additionally disables application-level asynchronous rendering,
  while windowed XR keeps it enabled for the spectator swapchain.
* Fixed headless XR recording startup by removing the explicit Kit visualizer
  option that conflicts with deprecated ``--headless`` handling. XR still
  auto-injects the Kit visualizer required for app-update pumping.
* Isolated windowed XR camera recording from CUDA error 700 failures by running
  single-environment PhysX simulation on CPU while retaining RTX, sensor
  rendering, CloudXR, and the spectator window on the selected display GPU.

Changed
^^^^^^^

* Changed windowed recording to physical GPU 0, which owns the workstation
  presentation queue, and headless recording to physical GPU 1. Physical GPUs
  2 and 3 remain quarantined after repeated driver-level Xid failures.
