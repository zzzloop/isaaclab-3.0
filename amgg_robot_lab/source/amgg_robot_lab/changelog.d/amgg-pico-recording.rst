Fixed
^^^^^

* Fixed PICO demonstration recording when the XR client supplied controller
  actions without sending an explicit ``START`` control event.
* Fixed AMGG XR RGB demonstration recording startup by disabling asynchronous
  multi-GPU rendering and using lightweight camera anti-aliasing.
* Fixed AMGG GPU selection so stale ``CUDA_VISIBLE_DEVICES`` settings are
  cleared before Isaac Sim RTX/Vulkan device discovery starts.
