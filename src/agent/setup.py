from setuptools import find_packages, setup
import os
from glob import glob
package_name = 'agent'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (f'share/{package_name}/launch', glob('launch/*.launch.py')),
        # (os.path.join('share', package_name, 'launch/nav2_bringup'), glob('launch/nav2_bringup/*.py')),
        # (os.path.join('share', package_name, 'params'), glob('params/*.yaml')),
        # (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # (os.path.join('share', package_name, 'worlds'), glob('worlds/*.world')),
        # (os.path.join('share', package_name, 'models/turtlebot3_burger'), glob('models/turtlebot3_burger/*')),
        # (os.path.join('share', package_name, 'models/turtlebot3_waffle'), glob('models/turtlebot3_waffle/*')),
        # (os.path.join('share', package_name, 'models/turtlebot3_waffle_pi'), glob('models/turtlebot3_waffle_pi/*')),
        (os.path.join('share', package_name, 'config'), glob('config/*')),
        # (os.path.join('share', package_name, 'urdf'), glob('urdf/*')),
        # (os.path.join('share', package_name, 'map'), glob('map/*')),

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
            # 'llm_nav = multi_robots.llm_nav:main',
            #'chat_gui = agent.chat_gui:main',
            'pick_place=agent.skill_server:main',
            'multi_object_detection = agent.multi_object_detection:main',
            # 'object_tracking = multi_robots.object_tracking_modified:main',  
            'planner= agent.Multi_robot_planner:main',
            'central_manager= agent.central_manager:main',
            'local_llm= agent.robot_local_llm:main',
            'agent= agent.agent_code:main',
            
            
            
        ],
    },
)

