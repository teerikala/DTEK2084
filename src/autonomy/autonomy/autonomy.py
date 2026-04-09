import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

custom_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

class Autonomy(Node):
    def __init__(self):
        super().__init__('autonomy')
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, custom_qos)
        self.bridge = CvBridge()
        self.cmd_pub = self.create_publisher(Twist, '/control', custom_qos)
        self.tello_client = self.create_client(TelloAction, '/tello_action')
        
        self.timer = self.create_timer(0.1, self.control_loop)
        self.x_err = 0
        self.y_err = 0
        #self.n = 0
        self.good_frames = 0
        self.speed = 0.0
        self.frames_going_forward = 0

        self.tello_service_call('takeoff')
        
    def tello_service_call(self, command):
        while not self.tello_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting")
        
        self.req = TelloAction.Request()
        self.req.cmd = command

        self.future = self.tello_client.call_async(self.req)
        self.future.add_done_callback(self.tello_callback)
        self.get_logger().info(f"Gave command: {command}")
 

    def tello_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")

    def control_loop(self):
        msg = Twist()
        
        msg.linear.x = self.speed
        msg.linear.y = -0.005 * self.x_err
        msg.linear.z = -0.005 * self.y_err

        self.cmd_pub.publish(msg)

    def image_callback(self, msg):
        # When enough good frames, go forward for ~30 sec, then land
        if self.good_frames > 20:
            if self.frames_going_forward < 1800:
                self.speed = 0.005
                self.frames_going_forward += 1
            else:
                self.frames_going_forward = 0
                self.good_frames = 0
                
                self.tello_service_call('land')

                return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        #cv2.imwrite(f"frame_{self.n}.png", image)
        #self.n += 1
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([30, 45, 45])
        upper_green = np.array([90, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((25, 25), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        #cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
        #rect = cv2.minAreaRect(largest)
        #box = cv2.boxPoints(rect)
        #box = np.int32(box)

        #(cx, cy) = rect[0]
        #cx, cy = int(cx), int(cy)
        M = cv2.moments(mask)

        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
        else:
            return

        frame_cx = image.shape[1] // 2
        frame_cy = image.shape[0] // 2-250
        
        cv2.circle(output, (frame_cx, frame_cy), 10, (255, 0, 0), -1)

        error_x = cx - frame_cx
        error_y = cy - frame_cy

        cv2.circle(output, (cx, cy), 10, (0, 0, 255), -1)
        #cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
        resized = output
        #resized = cv2.resize(output, None, fx=0.2, fy=0.2) 
        cv2.imshow("Rectangle", resized)
        cv2.waitKey(1)

        self.x_err = error_x
        self.y_err = error_y

        if error_x < 10 and error_y < 10:
            self.good_frames += 1
            self.get_logger().info(f"{self.good_frames} good frames")
        else:
            self.good_frames = 0
            self.get_logger().info("Bad frame, reset counter")

def main(args=None):
    rclpy.init(args=args)
    autonomy = Autonomy()
    rclpy.spin(autonomy)
    autonomy.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
