from setuptools import find_packages, setup

package_name = "panda_ik_window"

setup(
    name=package_name,
    version="0.1.0",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/ik_benchmark.launch.py",
                "launch/ik_self_collision_check.launch.py",
                "launch/ik_collision_pair_playback_rviz.launch.py",
            ],
        ),
        ("share/" + package_name + "/rviz", ["rviz/collision_pair_playback.rviz"]),
        # moveit_cpp.xml
        ("share/" + package_name + "/config", ["config/moveit_cpp_offline.yaml"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="MoveIt2 (MoveItPy) Panda IK dataset generation + window policy evaluation (configurable ws).",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ik_window = panda_ik_window.scripts.run_benchmark:main",
            "ik_check_self_collision = panda_ik_window.scripts.check_dataset_self_collision:main",
            "ik_collision_pair_player = panda_ik_window.scripts.collision_pair_player:main",
        ],
    },
)
