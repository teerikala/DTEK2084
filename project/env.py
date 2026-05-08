import matplotlib.pyplot as plt


class Env():
    def __init__(self):
        self.clock = 0
        self.robot_pos = [(-15.0, -15.0), (15.0, -15.0), (0.0, -15.0)]
        self.target_pos = [(0.0, 15.0)]
        
        # visualizer
        plt.ion() 
        self.fig, self.ax = plt.subplots(figsize=(8, 8))
        
        self.robot_scatter = self.ax.scatter([], [], c='blue', marker='o', s=80, 
                                             label='Robots', edgecolors='black', zorder=3)
        self.target_scatter = self.ax.scatter([], [], c='red', marker='x', s=100, 
                                              label='Targets', zorder=2)
        
        self.ax.set_title("Environment")
        self.ax.set_xlim((-20, 20))
        self.ax.set_ylim((-20, 20))
        self.ax.set_aspect('equal')
        self.ax.grid(True, linestyle='--', alpha=0.6)
        self.ax.legend(loc='upper right')
    
    def tick(self):
        self.clock += 1
        print(self.clock)
        
        self.update()

    def update(self):
        self.robot_scatter.set_offsets(self.robot_pos)
        self.target_scatter.set_offsets(self.target_pos)
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        plt.pause(0.01)

