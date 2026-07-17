Added
^^^^^

* Added three Unitree G1 with RH56DFX research tasks with PICO teleoperation,
  automatic evaluation, contact-force observations, and LeRobot conversion.
* Added 60 Hz XR variants of all three G1 tasks for lower-latency teleoperation.
* Added physical-GPU selection that defaults AMGG simulation, Kit rendering,
  and CloudXR to server GPU 2 while excluding the unreliable GPU 3.

Changed
^^^^^^^

* Changed the G1 tasks to use a clean experimental table, a publication-ready
  viewer and overview-camera angle, and a consistent 30 Hz control and camera
  sampling rate.
* Changed dynamic-object contact properties and masses to prevent excessive
  depenetration impulses while preserving gravity and stable grasping.
* Changed G1 XR contact simulation to use 240 Hz physics, bounded arm and hand
  drives, larger contact margins, and stronger constraint solving to prevent
  fast teleoperation commands from tunneling through task objects.

Fixed
^^^^^

* Fixed G1 task loading by copying the official robot configuration from a
  scene configuration instance.
