import sys

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.widgets import Button, CheckButtons
import tkinter as tk
from tkinter import filedialog
from pylab import mpl
import numpy as np
import pickle
import os
from simple_geometry import Point2D, Line2D
from simple_playground import Car, Playground

mpl.rcParams['font.sans-serif'] = ['Microsoft JhengHei']
mpl.rcParams['axes.unicode_minus'] = False


class CarSimulationGUI:
    def __init__(self):
        # Create figure and axes
        self.fig, self.ax = plt.subplots(figsize=(10, 8))
        self.fig.subplots_adjust(bottom=0.2)

        # Create playground and car
        self.playground = Playground()
        self.car = self.playground.car

        # For storing car path
        self.path_x = []
        self.path_y = []

        # Add buttons
        self.add_buttons()
        self.stop_on_collision = CheckButtons(plt.axes([0.01, 0.05, 0.15, 0.1]), ['碰撞時停下'], [False])
        self.train = CheckButtons(plt.axes([0.01, 0.35, 0.15, 0.1]), ['更改權重'], [True])

        # Initialize the visualization
        self.init_visualization()

        # Setup timer for animation
        self.timer = self.fig.canvas.new_timer(interval=10)
        self.timer.add_callback(self.update)

        # Flag to control simulation
        self.running = False

        # Show the plot
        plt.show()

    def add_buttons(self):
        ax_start = plt.axes([0.2, 0.05, 0.1, 0.075])
        ax_stop = plt.axes([0.35, 0.05, 0.1, 0.075])
        ax_reset = plt.axes([0.5, 0.05, 0.1, 0.075])
        ax_save = plt.axes([0.65, 0.05, 0.1, 0.075])
        ax_load = plt.axes([0.8, 0.05, 0.1, 0.075])

        self.btn_start = Button(ax_start, 'Start')
        self.btn_stop = Button(ax_stop, 'Stop')
        self.btn_reset = Button(ax_reset, 'Reset')
        self.btn_save = Button(ax_save, 'Save Q-Table')
        self.btn_load = Button(ax_load, 'Load Q-Table')

        self.btn_start.on_clicked(self.start_simulation)
        self.btn_stop.on_clicked(self.stop_simulation)
        self.btn_reset.on_clicked(self.reset_simulation)
        self.btn_save.on_clicked(self.save_qtable)
        self.btn_load.on_clicked(self.load_qtable)

    def init_visualization(self):
        # Clear plot
        self.ax.clear()

        # Draw playground boundaries
        self.draw_playground()

        # Draw initial car position
        self.car_body = self.draw_car()

        # Display car information
        self.car_info = self.display_car_info()

        # Draw path line (initially empty)
        self.path_line, = self.ax.plot([], [], 'r-', linewidth=2)

        # Set axis limits and labels
        self.ax.set_xlim(-10, 35)
        self.ax.set_ylim(-10, 55)
        self.ax.set_aspect('equal')
        self.ax.set_title('Car Simulation')
        self.ax.grid(False)

        # Force redraw
        self.fig.canvas.draw_idle()

    def draw_playground(self):
        # Draw all wall lines
        for line in self.playground.lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'b-', linewidth=2)

        # Draw destination line
        dest_line = self.playground.destination_line
        rect = Rectangle((dest_line.p1.x, dest_line.p1.y), dest_line.length, 0.5, angle=0, color='red', alpha=0.5)
        self.ax.add_patch(rect)

        # Draw decorative lines
        for line in self.playground.decorate_lines:
            self.ax.plot([line.p1.x, line.p2.x], [line.p1.y, line.p2.y], 'k-', linewidth=1)

        # Add text labels for important positions
        self.ax.text(-6, -3, '(-6, -3)', fontsize=8, ha='right')
        self.ax.text(6, -3, '(6, -3)', fontsize=8, ha='left')
        self.ax.text(-6, 0, '(-6, 0)', fontsize=8, ha='right')
        self.ax.text(6, 0, '(6, 0)', fontsize=8, ha='left')
        self.ax.text(6, 10, '(6, 10)', fontsize=8, ha='left')
        self.ax.text(18, 22, '(18, 22)', fontsize=8, ha='left')
        self.ax.text(-6, 22, '(-6, 22)', fontsize=8, ha='right')
        self.ax.text(30, 10, '(30, 10)', fontsize=8, ha='left')
        self.ax.text(30, 50, '(30, 50)', fontsize=8, ha='left')
        self.ax.text(18, 50, '(18, 50)', fontsize=8, ha='right')
        self.ax.text(24, 37, '終點線(18, 40)-(30, 37)', fontsize=8, ha='center', color='red')

    def draw_car(self):
        # Get car position and direction
        center_pos = self.car.getPosition('center')
        front_pos = self.car.getPosition('front')
        right_pos = self.car.getPosition('right')
        left_pos = self.car.getPosition('left')
        wheel_pos = self.car.getWheelPosPoint()

        # Draw car body (circle)
        car_body = plt.Circle((center_pos.x, center_pos.y), self.car.radius / 2, fill=False, color='red')
        self.ax.add_patch(car_body)

        # Draw direction line from center to front
        self.ax.plot([center_pos.x, front_pos.x], [center_pos.y, front_pos.y], 'k-')

        # Draw wheel direction line
        self.ax.plot([center_pos.x, wheel_pos.x], [center_pos.y, wheel_pos.y], 'g-')

        # Return car body for updates
        return car_body

    def display_car_info(self):
        return [
            self.ax.text(20, 0, f'車子中心: {self.car.getPosition("center"):.2f}', fontsize=8, ha='center',
                         color='green'),
            self.ax.text(20, 2, f'車子前方: {self.playground.state[0]:.2f}', fontsize=8, ha='center',
                         color='green'),
            self.ax.text(20, 4, f'車子右方: {self.playground.state[1]:.2f}', fontsize=8, ha='center',
                         color='green'),
            self.ax.text(20, 6, f'車子左方: {self.playground.state[2]:.2f}', fontsize=8, ha='center',
                         color='green'),
            self.ax.text(20, 8, f'方向盤角度: 0', fontsize=8, ha='center', color='green'),
        ]

    def save_qtable(self, event):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.asksaveasfilename(defaultextension=".pkl", filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()

        if filepath:
            try:
                self.playground.q_learner.save_qtable(filepath)
                print(f"Q-Table 已儲存為 {filepath}")
            except Exception as e:
                print(f"儲存失敗: {e}")

    def load_qtable(self, event):
        root = tk.Tk()
        root.withdraw()
        filepath = filedialog.askopenfilename(filetypes=[("Pickle Files", "*.pkl")])
        root.destroy()

        if filepath and os.path.exists(filepath):
            try:
                self.playground.q_learner.load_qtable(filepath)
                print(f"Q-Table 已載入: {filepath}")
                self.reset_simulation(None)
                if self.train.get_status()[0]:
                    self.train.set_active(0)
            except Exception as e:
                print(f"載入失敗: {e}")
        else:
            print("未選擇檔案或檔案不存在。")

    def update_car_visualization(self, action):
        # Update car position
        center_pos = self.car.getPosition('center')
        front_pos = self.car.getPosition('front')
        wheel_pos = self.car.getWheelPosPoint()

        # Update car body position
        self.car_body.center = (center_pos.x, center_pos.y)

        # Remove old direction lines
        for line in self.ax.lines:
            if line != self.path_line:  # Keep the path line
                line.remove()

        # Add updated direction lines
        self.ax.plot([center_pos.x, front_pos.x], [center_pos.y, front_pos.y], 'k-')
        self.ax.plot([center_pos.x, wheel_pos.x], [center_pos.y, wheel_pos.y], 'g-')

        # Draw playground boundaries again (they were removed by removing lines)
        self.draw_playground()

        # Update car info text
        self.car_info[0].set_text(f'車子中心: {center_pos:.2f}')
        self.car_info[1].set_text(f'車子前方: {self.playground.state[0]:.2f}')
        self.car_info[2].set_text(f'車子右方: {self.playground.state[1]:.2f}')
        self.car_info[3].set_text(f'車子左方: {self.playground.state[2]:.2f}')
        self.car_info[4].set_text(f'方向盤角度: {action:.2f}')
        if self.playground.state[1] - self.playground.state[2] > 0:
            self.car_info[4].set_color('green' if action > 0 else 'red')
        elif self.playground.state[1] - self.playground.state[2] < 0:
            self.car_info[4].set_color('red' if action > 0 else 'green')

        # Update path
        self.path_x.append(center_pos.x)
        self.path_y.append(center_pos.y)
        self.path_line.set_data(self.path_x, self.path_y)

    def update(self):
        if self.running and not self.playground.done:
            action = self.playground.predictAction(self.playground.state)
            state = self.playground.step(action, self.train.get_status()[0])
            self.update_car_visualization(action)
            self.fig.canvas.draw_idle()

        elif self.playground.done:
            if self.playground.isAtDestination():
                print("Car reached destination!")
                self.stop_simulation(None)
            else:
                print("Car crashed!")
                if self.stop_on_collision.get_status()[0]:  # 勾選「碰撞時停下」
                    self.stop_simulation(None)
                else:
                    self.reset_simulation(None)
                    self.start_simulation(None)

    def start_simulation(self, event):
        self.running = True
        self.timer.start()

    def stop_simulation(self, event):
        self.running = False
        self.timer.stop()

    def reset_simulation(self, event):
        self.stop_simulation(event)
        self.playground.reset()
        self.path_x = []
        self.path_y = []
        self.path_line.set_data(self.path_x, self.path_y)
        self.init_visualization()

    def save_path(self, event):
        if not self.path_x:
            print("No path to save!")
            return

        path_data = {
            'x': self.path_x,
            'y': self.path_y,
            'initial_position': (self.playground.car_init_pos.x, self.playground.car_init_pos.y)
            if self.playground.car_init_pos else (0, 0),
            'initial_angle': self.playground.car_init_angle if self.playground.car_init_angle else 90
        }

        # Save to file
        with open('car_path.pkl', 'wb') as f:
            pickle.dump(path_data, f)

        print("Path saved successfully!")

    def load_path(self, event):
        if not os.path.exists('car_path.pkl'):
            print("No saved path found!")
            return

        # Load from file
        with open('car_path.pkl', 'rb') as f:
            path_data = pickle.load(f)

        # Reset simulation
        self.reset_simulation(event)

        # Set loaded path
        self.path_x = path_data['x']
        self.path_y = path_data['y']
        self.path_line.set_data(self.path_x, self.path_y)

        # Set initial position if available
        if 'initial_position' in path_data:
            init_pos = Point2D(*path_data['initial_position'])
            init_angle = path_data.get('initial_angle', 90)
            self.playground.setCarPosAndAngle(init_pos, init_angle)
            self.update_car_visualization()

        print("Path loaded successfully!")
        self.fig.canvas.draw_idle()


def resource_path(self, relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        basePath = sys._MEIPASS
    except Exception:
        basePath = os.path.abspath(".")

    return os.path.join(basePath, relative_path)


if __name__ == "__main__":
    gui = CarSimulationGUI()
