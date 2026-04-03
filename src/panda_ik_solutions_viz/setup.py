from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'panda_ik_solutions_viz'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'config'), glob('config/*.yaml') + glob('config/*.yml') + glob('config/*.json')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='TODO',
    maintainer_email='TODO@example.com',
    description='Visualize multiple IK solutions for the Franka Panda in RViz by publishing a DisplayTrajectory containing all IK solutions.',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'display_ik_solutions = panda_ik_solutions_viz.display_ik_solutions:main',
        ],
    },
)
