Fixed
^^^^^

* Fixed PICO demonstration recording when the XR client supplied controller
  actions without sending an explicit ``START`` control event.
* Fixed AMGG XR RGB demonstration recording startup by disabling asynchronous
  multi-GPU rendering and using lightweight camera anti-aliasing.
* Fixed AMGG GPU selection so the preferred physical GPU is exposed as the only
  visible GPU before Isaac Sim starts.
