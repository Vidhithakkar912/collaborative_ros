"""PLACE routine. Input: box (x, y, z) in the arm base frame. No perception here."""

import math


from agent.arm_motion import ArmMotion, GRIPPER_OPEN, run_steps


PLACE_DROP_HEIGHT = 0.08
PLACE_PRE_HEIGHT = 0.15


class Placer:
    def __init__(self, arm: ArmMotion):
        self.arm = arm
        self.log = arm.node.get_logger()

    def place(self, x, y, z) -> bool:
        z_drop = z + PLACE_DROP_HEIGHT
        z_pre = z_drop + PLACE_PRE_HEIGHT
        self.log.info(
            f"PLACE in '{self.arm.arm_base_frame}': x={x:.3f} y={y:.3f} "
            f"z_drop={z_drop:.3f} z_pre={z_pre:.3f} | radial={math.hypot(x, y):.3f} m")

        return run_steps(self.log, [
            ("Pre-place (OMPL)", lambda: self.arm.move_to(x, y, z_pre)),
            ("Drop pose (straight down)",
             lambda: self.arm.move_straight(x, y, z_drop, avoid_collisions=False)),
            ("Release", lambda: self.arm.set_gripper(GRIPPER_OPEN)),
            ("Retreat (straight up)",
             lambda: self.arm.move_straight(x, y, z_pre, avoid_collisions=False)),
        ], self.arm.cancel_event)