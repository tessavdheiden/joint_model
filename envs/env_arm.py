"""
Environment for Robot Arm.
You can customize this script in a way you want.
View more on [莫烦Python] : https://morvanzhou.github.io/tutorials/
Requirement:
pyglet >= 1.2.4
numpy >= 1.12.1
"""
import numpy as np
import pyglet
import gym
from gym import spaces

#pyglet.clock.set_fps_limit(10000)


class ArmEnv(gym.Env):
    action_bound = np.array([1, 1])
    action_dim = 2
    state_dim = 6
    dt = .1  # refresh rate
    arm1l = 100
    arm2l = 100
    viewer = None
    viewer_xy = (400, 400)
    get_point = False
    mouse_in = np.array([False])
    point_l = 15
    grab_counter = 0
    bar_thc = 5

    def __init__(self, mode='easy'):
        # node1 (d_rad, x, y),
        # node2 (d_rad, x, y)
        self.mode = mode
        self.action_space = spaces.Box(low=self.action_bound, high=self.action_bound)
        self.observation_space = spaces.Box(low=np.zeros(self.state_dim), high=np.ones(self.state_dim))
        self.arm_info = np.zeros((2, 3))
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2

    def is_collision(self, p1, p2):
        delta_pos = p1 - p2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = self.bar_thc * 2
        return True if dist < dist_min else False

    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, -self.action_bound, self.action_bound)

        arm1rad = self.arm_info[0, 0]
        arm2rad = self.arm_info[1, 0]

        arm1rad += action[0] * self.dt
        arm2rad += action[1] * self.dt

        arm1dx_dy = np.array([self.arm1l * np.cos(arm1rad), self.arm1l * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm2l * np.cos(arm2rad), self.arm2l * np.sin(arm2rad)])
        arm1xy = self.center_coord + arm1dx_dy  # (x1, y1)
        arm2xy = self.arm_info[0, 1:3] + arm2dx_dy  # (x2, y2)
        self.arm_info[0, 0:3] = np.hstack([arm1rad, arm1xy[0], arm1xy[1]])
        if not self.is_collision(arm2xy, self.center_coord):
            self.arm_info[1, 0:3] = np.hstack([arm2rad, arm2xy[0], arm2xy[1]])

        s, arm2_distance = self._get_state()
        r = self._r_func(arm2_distance)

        return s, r, self.get_point

    def _reset_arm(self):
        arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
        self.arm_info[0, 0] = arm1rad
        self.arm_info[1, 0] = arm2rad
        arm1dx_dy = np.array([self.arm1l * np.cos(arm1rad), self.arm1l * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm2l * np.cos(arm2rad), self.arm2l * np.sin(arm2rad)])
        self.arm_info[0, 1:3] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[1, 1:3] = self.arm_info[0, 1:3] + arm2dx_dy  # (x2, y2)

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy
        else:
            self.point_info[:] = self.point_info_init

        self._reset_arm()
        while self.is_collision(self.arm_info[1, -2:], self.center_coord):
            self._reset_arm()

        return self._get_state()[0]

    def render(self):
        if self.viewer is None:
            self.viewer = Viewer(*self.viewer_xy, self.arm_info, self.point_info, self.point_l, self.mouse_in, self.bar_thc)
        self.viewer.render()

    def sample_action(self):
        return np.random.uniform(*self.action_bound, size=self.action_dim)

    def set_fps(self, fps=30):
        pyglet.clock.set_fps_limit(fps)

    def _get_state(self):
        # return the distance (dx, dy) between arm finger point with blue point
        arm_end = np.vstack([self.arm_info[0, 1:3], self.arm_info[1, 1:3]])
        t_arms = np.ravel(arm_end - self.point_info)
        center_dis = (self.center_coord - self.point_info)/200
        in_point = 1 if self.grab_counter > 0 else 0
        return np.hstack([t_arms/200, center_dis,
                          # arm1_distance_p, arm1_distance_b,
                          ]), t_arms[-2:]

    def _r_func(self, distance):
        t = 50
        abs_distance = np.sqrt(np.sum(np.square(distance)))
        r = -abs_distance/200
        if abs_distance < self.point_l and (not self.get_point):
            r += 1.
            self.grab_counter += 1
            if self.grab_counter > t:
                r += 10.
                self.get_point = True
        elif abs_distance > self.point_l:
            self.grab_counter = 0
            self.get_point = False
        return r


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
#    fps_display = pyglet.clock.ClockDisplay()

    def __init__(self, width, height, arm_info, point_info, point_l, mouse_in, bar_thc):
        super(Viewer, self).__init__(width, height, resizable=False, caption='Arm', vsync=False)  # vsync=False to not use the monitor FPS
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.arm_info = arm_info
        self.point_info = point_info
        self.mouse_in = mouse_in
        self.point_l = point_l
        self.bar_thc = bar_thc

        self.center_coord = np.array((min(width, height)/2, ) * 2)
        self.batch = pyglet.graphics.Batch()

        arm1_box, arm2_box, point_box = [0]*8, [0]*8, [0]*8
        c1, c2, c3 = (249, 86, 86)*4, (86, 109, 249)*4, (249, 39, 65)*4
        self.point = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', point_box), ('c3B', c2))
        self.arm1 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm1_box), ('c3B', c1))
        self.arm2 = self.batch.add(4, pyglet.gl.GL_QUADS, None, ('v2f', arm2_box), ('c3B', c1))

    def render(self):
        pyglet.clock.tick()
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[0, 1:3]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[0, 1:3]), *(self.arm_info[1, 1:3]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0, 0]
        x01, y01 = arm1_coord[0] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] + np.sin(
            arm1_thick_rad) * self.bar_thc
        x02, y02 = arm1_coord[0] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[1] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x11, y11 = arm1_coord[2] + np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] - np.sin(
            arm1_thick_rad) * self.bar_thc
        x12, y12 = arm1_coord[2] - np.cos(arm1_thick_rad) * self.bar_thc, arm1_coord[3] + np.sin(
            arm1_thick_rad) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[1, 0]
        x11_, y11_ = arm2_coord[0] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] - np.sin(
            arm2_thick_rad) * self.bar_thc
        x12_, y12_ = arm2_coord[0] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[1] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x21, y21 = arm2_coord[2] - np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] + np.sin(
            arm2_thick_rad) * self.bar_thc
        x22, y22 = arm2_coord[2] + np.cos(arm2_thick_rad) * self.bar_thc, arm2_coord[3] - np.sin(
            arm2_thick_rad) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0, 0] += .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0, 0] -= .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[1, 0] += .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[1, 0] -= .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.Q:
            pyglet.clock.set_fps_limit(1000)
        elif symbol == pyglet.window.key.A:
            pyglet.clock.set_fps_limit(30)

    def on_mouse_motion(self, x, y, dx, dy):
        self.point_info[:] = [x, y]

    def on_mouse_enter(self, x, y):
        self.mouse_in[0] = True

    def on_mouse_leave(self, x, y):
        self.mouse_in[0] = False
