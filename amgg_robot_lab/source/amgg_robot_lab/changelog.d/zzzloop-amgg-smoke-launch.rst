Fixed
^^^^^

* Fixed the AMGG smoke test startup order so that Kit initializes USD schema
  bindings before task and simulation modules are imported.
* Fixed the AMGG teleoperation and demonstration-recording wrappers to register
  custom tasks through the official post-AppLauncher callback.
* Fixed TCP observations and success terms when PhysX omits fixed virtual tool
  frames from the articulation body list.
