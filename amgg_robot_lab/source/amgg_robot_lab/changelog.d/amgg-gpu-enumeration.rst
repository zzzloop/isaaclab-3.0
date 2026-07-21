Fixed
^^^^^

* Fixed multi-GPU recording crashes caused by inconsistent CUDA and Kit/Vulkan
  device enumeration.
* Fixed XR demonstration recording using multi-GPU rendering by default, which
  could trigger CUDA peer-copy and external-memory failures on multi-GPU hosts.
* Fixed ``amgg_record_demos.py`` aborting with ``SystemExit`` on machines whose
  physical GPU inventory does not include the configured ``AMGG_PREFERRED_GPU``
  (e.g. single-GPU hosts). :func:`~amgg_robot_lab.scripts.amgg_gpu.configure_preferred_gpu`
  now falls back to the first available allowed GPU (then to the lowest
  available physical GPU) with a warning, so CUDA, Kit, and CloudXR are pinned
  to the same logical index instead of diverging across devices.
