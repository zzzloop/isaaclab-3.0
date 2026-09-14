# AM-DP123 asset data

`AM-DP123.urdf` and the 61 `meshes/*.STL` files are copied **byte-identically**
from the upstream AM-DP123 description package that was handed over with the
PICO XR integration request (see `AM_DP123_PICO_XR_HANDOFF.md`). No name
rewriting, mesh path rewriting, or limit editing is applied, so the files stay
auditable against the original SolidWorks export.

The URDF references its geometry as `package://AM-DP123/meshes/<name>.STL`. The
directory name `AM-DP123` therefore matches the ROS package name, which lets
`UrdfConverterCfg.ros_package_paths` resolve the meshes without editing the URDF.

Layout:

    AM-DP123/
    ├── urdf/AM-DP123.urdf      # 61 links, 60 joints (31 movable)
    └── meshes/*.STL            # 61 meshes, 63.4 MB

The URDF importer writes its generated USD next to the URDF
(`urdf/AM-DP123/AM-DP123.usda` and payloads). Those files are build artifacts and
are ignored by git; the first spawn in Isaac Sim regenerates them.
