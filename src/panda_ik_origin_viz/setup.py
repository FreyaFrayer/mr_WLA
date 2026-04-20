from setuptools import find_packages, setup

package_name = "panda_ik_origin_viz"

setup(
    name=package_name,
    version="0.0.1",
    packages=find_packages(exclude=["test"]),
    data_files=[
        ("share/ament_index/resource_index/packages", ["resource/" + package_name]),
        ("share/" + package_name, ["package.xml"]),
        (
            "share/" + package_name + "/launch",
            [
                "launch/ik_origin_compare_rviz.launch.py",
            ],
        ),
        (
            "share/" + package_name + "/rviz",
            [
                "rviz/ik_origin_compare.rviz",
            ],
        ),
        ("share/" + package_name, ["README.md"]),
    ],
    install_requires=["setuptools"],
    zip_safe=True,
    maintainer="you",
    maintainer_email="you@example.com",
    description="RViz2 visualization for comparing Panda IK origin, window and planner-replay paths.",
    license="Apache-2.0",
    tests_require=["pytest"],
    entry_points={
        "console_scripts": [
            "ik_origin_compare_player = panda_ik_origin_viz.ik_origin_compare_player:main",
        ],
    },
)
