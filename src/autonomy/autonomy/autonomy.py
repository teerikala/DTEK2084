import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class Autonomy(Node):
    def __init__(self):
        super().__init__('autonomy')
        self.camera_subscription = self.create_subscription(Image, '/image_raw', self.vision, 10)
        self.bridge = CvBridge()
        #self.cmd_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        #self.timer = self.create_timer(0.1, self.control_loop)
        self.x_err = 0
        self.y_err = 0

    def control_loop(self):
        msg = Twist()
        
        msg.linear.x = -0.002 * self.x_err
        msg.linear.y = -0.002 * self.y_err

        self.cmd_pub.publish(msg)

    def vision(self, msg):
        self.get_logger().info('Send help to Samppa')
        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        self.get_logger().info('Frame received')
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

        lower_green = np.array([35, 50, 50])
        upper_green = np.array([85, 255, 255])

        mask = cv2.inRange(hsv, lower_green, upper_green)

        kernel = np.ones((25, 25), np.uint8)

        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            largest = max(contours, key=cv2.contourArea)
            #x, y, w, h = cv2.boundingRect(largest)

            output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
            #cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
            rect = cv2.minAreaRect(largest)
            box = cv2.boxPoints(rect)
            box = np.int32(box)

            (cx, cy) = rect[0]
            cx, cy = int(cx), int(cy)

            frame_cx = image.shape[1] // 2
            frame_cy = image.shape[0] // 2-250
            
            cv2.circle(output, (frame_cx, frame_cy), 10, (255, 0, 0), -1)

            error_x = cx - frame_cx
            error_y = cy - frame_cy

            cv2.circle(output, (cx, cy), 10, (0, 0, 255), -1)
            cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
            resized = cv2.resize(output, None, fx=0.2, fy=0.2) 
            cv2.imshow("Rectangle", resized)
            cv2.waitKey(0)
            cv2.destroyAllWindows()

            self.x_err = error_x
            self.y_err = error_y

def main():
    rclpy.init()
    node = Autonomy()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
