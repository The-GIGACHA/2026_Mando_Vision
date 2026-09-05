from distutils.core import setup
from catkin_pkg.python_setup import generate_distutils_setup

setup(**generate_distutils_setup(
    packages=['mando_vision_2026_utils'],
    package_dir={'mando_vision_2026_utils': 'utils'},
))
