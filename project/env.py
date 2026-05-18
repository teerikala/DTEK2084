import matplotlib.pyplot as plt
from matplotlib.patches import Wedge, Polygon
import numpy as np

def intersect_lines(p1, p2, p3, p4):
    x1, y1 = p1
    x2, y2 = p2
    x3, y3 = p3
    x4, y4 = p4

    denominator = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)

    if abs(denominator) < 1e-10:
        return None 

    px = ((x1*y2 - y1*x2)*(x3 - x4) - (x1 - x2)*(x3*y4 - y3*x4)) / denominator
    py = ((x1*y2 - y1*x2)*(y3 - y4) - (y1 - y2)*(x3*y4 - y3*x4)) / denominator

    return np.array([px, py])

def calculate_area(points):
    if len(points) < 3:
        return 0.0
    
    pts = np.array(points)
    x = pts[:, 0]
    y = pts[:, 1]
    
    # The Shoelace Formula
    area = 0.5 * np.abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    return area

def signal_strenght(sender_pos, receiver_pos, p_transmit=20, n=2.0):
    dist = np.linalg.norm(np.array(sender_pos) - np.array(receiver_pos))

    # Log-Distance Path Loss Model
    path_loss = 10 * n * np.log10(dist) + 35
    
    return p_transmit - path_loss

def angle_of_arrival(sender_pos, receiver_pos):
    dist = np.linalg.norm(np.array(sender_pos) - np.array(receiver_pos))

    # Angle of Arrival
    dx = sender_pos[0] - receiver_pos[0]
    dy = sender_pos[1] - receiver_pos[1]
    gt_angle = np.arctan2(dy, dx)

    # max_noise = base uncertainty + distance factor
    max_noise = np.radians(0.5 + (dist**2) * 0.00001)

    # total angle uncertainty = base uncertainty + noise
    angle_uncertainty = np.random.normal(0, max_noise)

    estimated_angle = gt_angle + angle_uncertainty

    # 95 % chance of target being within the limits
    return np.degrees(estimated_angle - 2.0*max_noise), np.degrees(estimated_angle + 2.0*max_noise)

class Env():
    def __init__(self, robot_pos=[], target_pos=[]):
        self.robot_pos = robot_pos
        self.target_pos = target_pos
        
        n_robots = len(self.robot_pos)
        n_targets = len(self.target_pos)
        
        self.clock = 0
        
        # visualizer
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.robot_scatter = self.ax.scatter([], [], c='blue', marker='o', s=80, 
                                             label='Robots', edgecolors='black', zorder=3)
        self.target_scatter = self.ax.scatter([], [], c='red', marker='x', s=100, 
                                              label='Targets', zorder=2)
                                              
        self.local_scatter = self.ax.scatter([], [], c='green', marker='.', s=50, 
                                             label='Raw Sensor Guesses', alpha=0.5, zorder=4)
        self.consensus_scatter = self.ax.scatter([], [], c='orange', marker='*', s=150, 
                                                 label='Consensus State', edgecolors='black', zorder=5)
        
        self.ax.set_title("Environment")
        self.ax.set_xlim((-200, 200))
        self.ax.set_ylim((-200, 200))
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend(loc='upper right')

        self.wedge_info = []
        self.wedges = []
        for i in range(n_robots*n_targets):
            self.wedge_info.append(None)
            self.wedges.append(None)
    
    def draw_beliefs(self, local_ests, consensus_ests):
        """Updates the plot with the robots' internal calculations."""
        if len(local_ests) > 0:
            self.local_scatter.set_offsets(local_ests)
        if len(consensus_ests) > 0:
            self.consensus_scatter.set_offsets(consensus_ests)
    
    def get_estimated_target(self):
        """Returns the center of the current uncertainty area, if valid."""
        if hasattr(self, 'area_patch') and self.area_patch is not None:
            # Get the corners of the current intersection polygon
            corners = self.area_patch.get_xy()
            if len(corners) > 0:
                mean_x = np.mean(corners[:, 0])
                mean_y = np.mean(corners[:, 1])
                return np.array([mean_x, mean_y])
        return None    
    
    def tick(self):
        self.clock += 1
        #print(self.clock)
        
        self.update()

    def update(self):
        i = 0
        for t_pos in self.target_pos:
            for r_pos in self.robot_pos:
                s = signal_strenght(t_pos, r_pos)
                t1, t2 = angle_of_arrival(t_pos, r_pos)
                self.wedge_info[i] = [r_pos, t1, t2]
                
                i += 1

        for r_pos1 in self.robot_pos:
            for r_pos2 in self.robot_pos:
                if r_pos1 == r_pos2:
                    continue
                
                s = signal_strenght(r_pos1, r_pos2)
                if s > -60.0:
                    #print(r_pos2, "receiving from", r_pos1)
                    pass
                
                #print(s)

        # visualizer
        self.robot_scatter.set_offsets(self.robot_pos)
        self.target_scatter.set_offsets(self.target_pos)

        left_borders = []
        right_borders = []
        robots_data = []

        for i in range(len(self.wedges)):
            if self.wedges[i]:
                self.wedges[i].remove()

            if not self.wedge_info[i]:
                continue
            
            wedge = self.wedge_info[i]
            self.wedges[i] = Wedge(wedge[0], 500.0, wedge[1], wedge[2], color='blue', alpha=0.3, label='Uncertainty')
            self.ax.add_patch(self.wedges[i])

            x1, y1 = wedge[0]
            
            angle_right = np.radians(wedge[1])
            angle_left = np.radians(wedge[2])

            v_right = (np.cos(angle_right), np.sin(angle_right))
            v_left = (np.cos(angle_left), np.sin(angle_left))

            right_borders.append([x1, y1, x1 + 300*v_right[0], y1 + 300*v_right[1]])
            left_borders.append([x1, y1, x1 + 300*v_left[0], y1 + 300*v_left[1]])

            robots_data.append({'pos': (x1, y1), 'v_right': v_right, 'v_left': v_left})


        intersections = []
        for b1 in right_borders + left_borders:
            for b2 in right_borders + left_borders:
                if b1[0] == b2[0] and b1[1] == b2[1]:
                    continue

                itx = intersect_lines((b1[0], b1[1]), (b1[2], b1[3]),
                                      (b2[0], b2[1]), (b2[2], b2[3]))
        
                if itx.any() == None:
                    continue

                is_valid = True
                for r in robots_data:
                    to_pt = (itx[0] - r['pos'][0], itx[1] - r['pos'][1])

                    cross_r = r['v_right'][0]*to_pt[1] - r['v_right'][1]*to_pt[0]
                    cross_l = r['v_left'][0]*to_pt[1] - r['v_left'][1]*to_pt[0]

                    if not (cross_r >= -1e-9 and cross_l <= 1e-9):
                        is_valid = False
                        break

                if is_valid:
                    intersections.append(itx)

        if len(intersections) >= 3:
            mean_x = np.mean([p[0] for p in intersections])
            mean_y = np.mean([p[1] for p in intersections])
            
            def get_angle(p):
                return np.arctan2(p[1] - mean_y, p[0] - mean_x)
            
            sorted_intersections = sorted(intersections, key=get_angle)

            area = calculate_area(sorted_intersections)
            print(round(area), "m²")

            if hasattr(self, 'area_patch'):
                self.area_patch.remove()

            self.area_patch = Polygon(sorted_intersections, 
                                      closed=True, 
                                      color='pink', 
                                      alpha=0.7, 
                                      edgecolor='deeppink', 
                                      linewidth=2)
            
            self.ax.add_patch(self.area_patch)

        elif hasattr(self, 'area_patch'):
            self.area_patch.remove()
            del self.area_patch

        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        plt.pause(0.01)
    
