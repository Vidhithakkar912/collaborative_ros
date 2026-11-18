"""PICK routine. Input: object (x, y, z) in the arm base frame. No perception here."""

import math


from agent.arm_motion import ArmMotion, GRIPPER_CLOSE, GRIPPER_OPEN, run_steps


GRASP_PRE_HEIGHT = -0.07
GRASP_LIFT_HEIGHT = 0.05
GRASP_Z_OFFSET = 0.0


class Picker:
    def __init__(self, arm: ArmMotion):
        self.arm = arm
        self.log = arm.node.get_logger()

    def pick(self, x, y, z, probe=True) -> bool:
        self.log.info(
            f"PICK in '{self.arm.arm_base_frame}': x={x:.3f} y={y:.3f} z={z:.3f} "
            f"| yaw={math.degrees(math.atan2(y, x)):.1f} deg "
            f"| radial={math.hypot(x, y):.3f} m")
        if probe:
            self.arm.probe_reach(x, y, z)

        return run_steps(self.log, [
            ("Open gripper", lambda: self.arm.set_gripper(GRIPPER_OPEN)),
            ("Pre-grasp (OMPL)", lambda: self.arm.move_to(x, y, z + GRASP_PRE_HEIGHT)),
            ("Settle + log", lambda: self.arm.settle_and_log((x, y, z + GRASP_Z_OFFSET))),
            ("Close gripper", lambda: self.arm.set_gripper(GRIPPER_CLOSE, closing=True)),
            # ("Lift (straight up)",
            #  lambda: self.arm.move_straight(x, y, z + GRASP_LIFT_HEIGHT,
            #                                 avoid_collisions=False)),
            ("Move to home pose", self.arm.move_home),
        ], self.arm.cancel_event)