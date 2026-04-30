import rclpy
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from geometry_msgs.msg import Twist
from tello_msgs.srv import TelloAction
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import time

custom_qos = QoSProfile(history=HistoryPolicy.KEEP_LAST, depth=10, reliability=ReliabilityPolicy.BEST_EFFORT, durability=DurabilityPolicy.VOLATILE)

class Autonomy(Node):
    def __init__(self):
        super().__init__('autonomy')
        self.subscription = self.create_subscription(Image, '/image_raw', self.image_callback, custom_qos)
        self.bridge = CvBridge()
        
        # Service client for discrete commands (Takeoff, Land)
        self.tello_client = self.create_client(TelloAction, '/tello_action')
        
        # Publisher for /cmd_vel
        self.cmd_vel_pub = self.create_publisher(Twist, '/cmd_vel', 10)
        
        self.x_err = 0
        self.y_err = 0
        #self.n = 0
        self.good_frames = 0
        self.speed = 0.0
        self.frames_going_forward = 0
        self.prev_cx = None
        self.prev_cy = None

        self.aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
        self.parameters = cv2.aruco.DetectorParameters_create()

        self.command_in_progress = False
        
        self.mission_state = 'TRACKING'
        
        self.reset_timer = None

        self.tello_service_call('takeoff')
        
    def tello_service_call(self, command):
    
        if self.command_in_progress:
            return
        
        while not self.tello_client.wait_for_service(timeout_sec=1.0):
            self.get_logger().info("Service not available, waiting")
        
        self.req = TelloAction.Request()
        self.req.cmd = command
        
        self.command_in_progress = True

        self.future = self.tello_client.call_async(self.req)
        self.future.add_done_callback(self.tello_callback)
        self.get_logger().info(f"Gave command: {command}")
 

    def tello_callback(self, future):
        try:
            response = future.result()
            self.get_logger().info(f"Response: {response}")
        except Exception as e:
            self.get_logger().error(f"Service call failed: {e}")
        finally:
            # Unlock the drone state once the command finishes
            self.command_in_progress = False

    def image_callback(self, msg):
    
        if self.mission_state != 'TRACKING' or self.command_in_progress:
           return

        image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8') 
        #cv2.imwrite(f"frame_{self.n}.png", image)
        #self.n += 1
        
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        corners, ids, rejected = cv2.aruco.detectMarkers(gray, self.aruco_dict, parameters=self.parameters)
        if ids is not None:
            for i in range(len(ids)):
                if ids[i][0] == 4:
                    cx, cy = np.mean(corners[i][0], axis=0)
                    cx, cy = int(cx)+20, int(cy)-50
                    self.prev_cx, self.prev_cy = cx, cy
        
        cx, cy = self.prev_cx, self.prev_cy
        
        current_width = 0
        target_found = False
        
        if not self.prev_cx and not self.prev_cy:

            hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

            lower_green = np.array([37, 47, 47])
            upper_green = np.array([90, 255, 255])
            
            #lower_green = np.array([30, 45, 45])
            #upper_green = np.array([90, 255, 255])

            mask = cv2.inRange(hsv, lower_green, upper_green)
            
            lower_red1 = np.array([0, 120, 100])
            upper_red1 = np.array([1, 255, 255])

            lower_red2 = np.array([170, 120, 100])
            upper_red2 = np.array([180, 255, 255])

            mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
            mask2 = cv2.inRange(hsv, lower_red2, upper_red2)

            kernel = np.ones((25, 25), np.uint8)
            
            red_mask = mask1 + mask2
            red_count = cv2.countNonZero(red_mask)
            
            self.get_logger().info(f"Red pixels: {red_count}")
            
            if red_count > 4000:
                self.get_logger().info("Beginning landing sequence...")
                self.cmd_vel_pub.publish(Twist())
                self.command_in_progress = True
                self.tello_service_call('land')
                time.sleep(3)
                return

            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

            output = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
            valid_contours = []
            total_gate_area = 0
         
            if contours:
                for c in contours:
                    area = cv2.contourArea(c)
                    
                    if area > 2000: # <- how many pixels need to be in an area to be count towards the total area (this is for ignoring interference)
                        valid_contours.append(c)
                        total_gate_area += area
                        
                if len(valid_contours) > 0 and total_gate_area > 5000: # <- how many pixels need to be in the total area to start adjusting position and aiming towards the area's middle point
                    target_found = True
                    
                    all_points = np.vstack(valid_contours)
                    
                    x, y, w, h = cv2.boundingRect(all_points)
                    
                    cx = x + (w // 2)
                    cy = y + (h // 2)
                    current_width = w
                    
                    cv2.rectangle(output, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.drawContours(output, valid_contours, -1, (0, 100, 0), 1)
                    
            if not target_found:
                self.get_logger().info("Target lost or too small! Searching right...")
                search_msg = Twist()
                search_msg.angular.z = -0.3 # The rotational speed to search gate
                self.cmd_vel_pub.publish(search_msg)
                
                self.good_frames = 0
                
                # Show the video feed so the window updates while spinning
                frame_cx = image.shape[1] // 2
                frame_cy = image.shape[0] // 2 - 100
                cv2.circle(output, (frame_cx, frame_cy), 10, (255, 0, 0), -1)
                cv2.imshow("Rectangle", output)
                cv2.waitKey(1)
                return
            else:
                self.cmd_vel_pub.publish(Twist())
            
            M = cv2.moments(mask)
            
            current_area = cv2.countNonZero(mask)

        frame_cx = image.shape[1] // 2
        frame_cy = image.shape[0] // 2 - 100
        
        output = image
        cv2.circle(output, (frame_cx, frame_cy), 10, (255, 0, 0), -1)

        error_x = cx - frame_cx
        error_y = cy - frame_cy
	
        cv2.circle(output, (cx, cy), 10, (0, 0, 255), -1)
        #cv2.drawContours(output, [box], 0, (0, 255, 0), 2)
        resized = output
        #resized = cv2.resize(output, None, fx=0.2, fy=0.2) 
        cv2.imshow("Rectangle", resized)
        cv2.waitKey(1)
        
        TARGET_WIDTH = 600
        
        # NOTE: Might crash to gate 2 !!!
        if (current_width > (TARGET_WIDTH * 0.90) and abs(error_x) <= 30 and abs(error_y) <= 30) or (self.prev_cx and self.prev_cy and abs(error_x) <= 25 and abs(error_y) <= 25):
            self.good_frames += 1
            if self.good_frames > 20:
                self.get_logger().info("AT THE GATE! Pushing through...")
                self.cmd_vel_pub.publish(Twist())
                self.mission_state = 'PASSING_GATE'
                self.tello_service_call('forward 225')
                self.good_frames = 0
                time.sleep(3)
                self.mission_state = 'TRACKING'
                self.prev_cx = None
                self.prev_cy = None
                return
            
            
        #self.x_err = error_x
        #self.y_err = error_y
        
        
        #if abs(error_x) <= 20 and abs(error_y) <= 20:
            #self.cmd_vel_pub.publish(Twist())
            #self.good_frames += 1
            #self.get_logger().info(f"{self.good_frames}/20 good frames")
        else:
            self.good_frames = 0
            velocity_h = 0
            twist_msg = Twist()

            # NOTE: Might crash to gate 2 !!!
            if self.prev_cx and self.prev_cy:
                Kp_r = 0.0008
                Kp_v = 0.003
            else:
                Kp_r = 0.004 		# Rotational speed for adjusting towards green gates
                Kp_v = 0.003 		# Vertical speed for adjusting towards green gates
                Kp_depth = 0.001 	# Horizontal speed for adjusting towards green gates
            
                error_width = TARGET_WIDTH - current_width
                velocity_h = float(error_width * Kp_depth)
            
                alignment_penalty = max(1.0, (abs(error_x) + abs(error_y)) / 50.0)
                velocity_h = velocity_h / alignment_penalty
                velocity_h = max(-0.3, min(0.3, velocity_h))
                twist_msg.linear.x = velocity_h

            velocity_r = -float(error_x * Kp_r)
            velocity_v = -float(error_y * Kp_v)
                        
            max_speed = 0.5
            velocity_r = max(-max_speed, min(max_speed, velocity_r))
            velocity_v = max(-max_speed, min(max_speed, velocity_v))
            
            twist_msg.angular.z = velocity_r
            twist_msg.linear.z = velocity_v
            
            self.cmd_vel_pub.publish(twist_msg)
            
            #self.get_logger().info(f"Width: {current_width} / {TARGET_WIDTH} | vel_h (fwd): {velocity_h:.2f}")
            
            #self.get_logger().info(f"Tracking -> vel_r: {velocity_r:.2f}, vel_v: {velocity_v:.2f}")
            
def main(args=None):
    rclpy.init(args=args)
    autonomy = Autonomy()
    
    try:
        rclpy.spin(autonomy)
    except KeyboardInterrupt:
        pass
    finally:
        #failsafe: stop drone on shutdown
        failsafe_msg = Twist()
        autonomy.cmd_vel_pub.publish(failsafe_msg)
        autonomy.destroy_node()
        rclpy.shutdown()

if __name__ == '__main__':
    main()
