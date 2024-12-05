import sys
import rclpy
from rclpy.node import Node
from rclpy.signals import SignalHandlerOptions
from rclpy.executors import ExternalShutdownException, MultiThreadedExecutor
from rclpy.qos import QoSPresetProfiles
from rclpy.callback_groups import MutuallyExclusiveCallbackGroup
from geometry_msgs.msg import Twist
from nav_msgs.msg import Odometry
from sensor_msgs.msg import LaserScan
from assessment_interfaces.msg import Item, ItemList
from tf_transformations import euler_from_quaternion
import angles
from enum import Enum
import random
import math

LINEAR_VELOCITY = 0.3  # Metres per second
ANGULAR_VELOCITY = 0.5  # Radians per second
SCAN_THRESHOLD = 0.5  # Metres per second

# Finite state machine (FSM) states
class State(Enum):
    FORWARD = 0
    TURNING = 1
    COLLECTING = 2

class RobotController(Node):
    def __init__(self):
        super().__init__('robot_controller')
        self.state = State.FORWARD
        self.pose = None
        self.previous_pose = None
        self.yaw = 0.0
        self.previous_yaw = 0.0
        self.turn_angle = 0.0
        self.turn_direction = 1  # Default to TURN_LEFT
        self.goal_distance = random.uniform(1.0, 2.0)
        self.scan_triggered = [False] * 4
        self.items = ItemList()

        self.declare_parameter('robot_id', 'robot1')
        self.robot_id = self.get_parameter('robot_id').value

        timer_callback_group = MutuallyExclusiveCallbackGroup()

        self.item_subscriber = self.create_subscription(ItemList, '/items', self.item_callback, 10, callback_group=timer_callback_group)
        self.odom_subscriber = self.create_subscription(Odometry, 'odom', self.odom_callback, 10, callback_group=timer_callback_group)
        self.scan_subscriber = self.create_subscription(LaserScan, 'scan', self.scan_callback, QoSPresetProfiles.SENSOR_DATA.value, callback_group=timer_callback_group)
        self.cmd_vel_publisher = self.create_publisher(Twist, 'cmd_vel', 10)

        self.timer_period = 0.1  # 100 milliseconds = 10 Hz
        self.timer = self.create_timer(self.timer_period, self.control_loop, callback_group=timer_callback_group)

    def item_callback(self, msg):
        self.items = msg

    def odom_callback(self, msg):
        self.pose = msg.pose.pose
        (roll, pitch, yaw) = euler_from_quaternion([self.pose.orientation.x, self.pose.orientation.y, self.pose.orientation.z, self.pose.orientation.w])
        self.yaw = yaw

    def scan_callback(self, msg):
        front_ranges = msg.ranges[331:359] + msg.ranges[0:30]
        left_ranges = msg.ranges[31:90]
        right_ranges = msg.ranges[271:330]

        self.scan_triggered[0] = min(front_ranges) < SCAN_THRESHOLD
        self.scan_triggered[1] = min(left_ranges) < SCAN_THRESHOLD
        self.scan_triggered[2] = min(right_ranges) < SCAN_THRESHOLD

    def control_loop(self):
        match self.state:
            case State.FORWARD:
                if self.scan_triggered[0]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = random.uniform(150, 170)
                    self.turn_direction = random.choice([1, -1])  # TURN_LEFT or TURN_RIGHT
                    self.get_logger().info("Detected obstacle in front, turning " + ("left" if self.turn_direction == 1 else "right") + f" by {self.turn_angle:.2f} degrees")
                    return

                if self.scan_triggered[1] or self.scan_triggered[2]:
                    self.previous_yaw = self.yaw
                    self.state = State.TURNING
                    self.turn_angle = 45
                    if self.scan_triggered[1] and self.scan_triggered[2 ]:
                        self.turn_direction = random.choice([1, -1])
                        self.get_logger().info("Detected obstacle to both the left and right, turning " + ("left" if self.turn_direction == 1 else "right") + f" by {self.turn_angle:.2f} degrees")
                    elif self.scan_triggered[1]:
                        self.turn_direction = -1  # TURN_RIGHT
                        self.get_logger().info(f"Detected obstacle to the left, turning right by {self.turn_angle} degrees")
                    else:
                        self.turn_direction = 1  # TURN_LEFT
                        self.get_logger().info(f"Detected obstacle to the right, turning left by {self.turn_angle} degrees")
                    return

                if len(self.items.data) > 0:
                    self.state = State.COLLECTING
                    return

                msg = Twist()
                msg.linear.x = LINEAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

            case State.TURNING:
                msg = Twist()
                msg.angular.z = self.turn_direction * ANGULAR_VELOCITY
                self.cmd_vel_publisher.publish(msg)

                yaw_difference = angles.normalize_angle(self.yaw - self.previous_yaw)

                if abs(yaw_difference) >= math.radians(self.turn_angle):
                    self.previous_pose = self.pose
                    self.goal_distance = random.uniform(1.0, 2.0)
                    self.state = State.FORWARD
                    self.get_logger().info(f"Finished turning, driving forward by {self.goal_distance:.2f} metres")

            case State.COLLECTING:
                if len(self.items.data) == 0:
                    self.previous_pose = self.pose
                    self.state = State.FORWARD
                    return

                item = self.items.data[0]
                estimated_distance = 32.4 * float(item.diameter) ** -0.75

                msg = Twist()
                msg.linear.x = 0.25 * estimated_distance
                self.cmd_vel_publisher.publish(msg)

            case _:
                pass

    def destroy_node(self):
        msg = Twist()
        self.cmd_vel_publisher.publish(msg)
        self.get_logger().info(f"Stopping: {msg}")
        super().destroy_node()

def main(args=None):
    rclpy.init(args=args, signal_handler_options=SignalHandlerOptions.NO)
    node = RobotController()
    executor = MultiThreadedExecutor()
    executor.add_node(node)

    try:
        executor.spin()
    except KeyboardInterrupt:
        pass
    except ExternalShutdownException:
        sys.exit(1)
    finally:
        node.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()