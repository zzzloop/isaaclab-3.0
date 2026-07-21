Fixed
^^^^^

* Fixed PICO demonstration recording when the XR client supplied controller
  actions without sending an explicit ``START`` control event.
* Fixed AMGG XR RGB demonstration recording startup by disabling multi-GPU
  rendering and using FXAA camera anti-aliasing.
* Changed AMGG XR recording to use the RTX Minimal full-material render mode
  for stable low-noise PICO teleoperation while recording camera observations.
* Fixed AMGG GPU selection so stale ``CUDA_VISIBLE_DEVICES`` settings are
  cleared before Isaac Sim RTX/Vulkan device discovery starts.
