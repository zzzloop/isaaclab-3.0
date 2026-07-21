Fixed
^^^^^

* Fixed PICO demonstration recording when the XR client supplied controller
  actions without sending an explicit ``START`` control event.
* Fixed AMGG XR RGB demonstration recording startup by disabling multi-GPU
  rendering while preserving the normal DLSS XR teleoperation view.
* Fixed AMGG demonstration collection to default to continuous recording so
  successful episodes reset and continue instead of exiting after one demo.
* Fixed AMGG GPU selection so stale ``CUDA_VISIBLE_DEVICES`` settings are
  cleared before Isaac Sim RTX/Vulkan device discovery starts.
