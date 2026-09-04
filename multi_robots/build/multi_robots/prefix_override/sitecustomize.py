import sys
if sys.prefix == '/usr':
    sys.real_prefix = sys.prefix
    sys.prefix = sys.exec_prefix = '/home/einfochips/ros2_ws/src/multi_robots/install/multi_robots'
