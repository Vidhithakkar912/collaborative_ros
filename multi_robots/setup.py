from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'multi_robots'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'launch/nav2_bringup'), glob('launch/nav2_bringup/*.py')),
        (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        (os.path.join('share', package_name, 'models/turtlebot3_burger'), glob('models/turtlebot3_burger/*')),
        (os.path.join('share', package_name, 'models/turtlebot3_waffle'), glob('models/turtlebot3_waffle/*')),
        (os.path.join('share', package_name, 'models/turtlebot3_waffle_pi'), glob('models/turtlebot3_waffle_pi/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        (os.path.join('share', package_name, 'map'), glob('map/*')),

    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='einfochips',
    maintainer_email='einfochips@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'llm_nav = multi_robots.llm_nav:main',
            'chat_gui = multi_robots.chat_gui:main',
            'object_detection = multi_robots.object_detection:main',
            'object_pick = multi_robots.object_pick:main',
            'multi_object_detection = multi_robots.multi_object_detection:main',
            'object_tracking = multi_robots.object_tracking_modified:main',
            'llm_obj=multi_robots.llm_search_object:main',
            'goal_pose=multi_robots.goal_pose:main',
            'llm_decision=multi_robots.llm_decision_making:main',
            'planner=multi_robots.robot_planner_multi_object:main',
            'task_manager=multi_robots.central_manager:main',
            'manipulator=multi_robots.manipulator_brain:main',
            'transporter=multi_robots.transporter_brain:main',
            'filter=multi_robots.filter:main',
            
            
        ],
    },
)

