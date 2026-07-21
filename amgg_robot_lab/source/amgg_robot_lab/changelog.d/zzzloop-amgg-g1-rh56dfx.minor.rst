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
* Changed all G1 task objects, goals, and fixtures to stay inside the observed
  0.42 m forward-reach limit with margin for randomized resets.
* Changed G1 success tolerances for practical PICO data collection and added
  visible success or failure feedback to the teleoperation loop.
* Changed the bimanual and precision task spawn layouts to remove initial
  object-to-fixture and object-to-table penetration.
* Changed the XR recording status to show an explicit per-demo success and
  reset message in the headset.
