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
import torch

#pyglet.clock.set_fps_limit(10000)


class ArmEnv(gym.Env):
    action_bound = np.array([1, 1])
    action_dim = 2
    state_dim = 8
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
        self.arm_info = np.zeros(8)
        self.point_info = np.array([250, 303])
        self.point_info_init = self.point_info.copy()
        self.center_coord = np.array(self.viewer_xy)/2

    def is_collision(self, p1, p2):
        delta_pos = p1 - p2
        dist = np.sqrt(np.sum(np.square(delta_pos)))
        dist_min = self.bar_thc * 3
        return True if dist < dist_min else False

    def step(self, action):
        # action = (node1 angular v, node2 angular v)
        action = np.clip(action, -self.action_bound, self.action_bound)

        arm1rad = self.arm_info[0]
        arm2rad = self.arm_info[3]

        arm1rad = (arm1rad + action[0] * self.dt) % (2 * np.pi)
        arm2rad = (arm2rad + action[1] * self.dt) % (2 * np.pi)

        arm1dx_dy = np.array([self.arm1l * np.cos(arm1rad), self.arm1l * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm2l * np.cos((arm1rad + arm2rad) % (2 * np.pi)), self.arm2l * np.sin((arm1rad + arm2rad) % (2 * np.pi))])
        arm1xy = self.center_coord + arm1dx_dy  # (x1, y1)
        arm2xy = arm1xy + arm2dx_dy  # (x2, y2)

        #if not self.is_collision(arm2xy, self.center_coord):
        self.arm_info[:3] = np.hstack([arm1rad, arm1xy[0], arm1xy[1]])
        self.arm_info[3:6] = np.hstack([arm2rad, arm2xy[0], arm2xy[1]])
        self.arm_info[6:8] = np.hstack([arm1rad, (arm1rad + arm2rad) % (2 * np.pi)])

        s, arm2_distance = self._get_state()
        r = self._r_func(arm2_distance)

        return s, r, self.get_point

    def step_batch(self, x, u):
        def is_collision(p1, p2):
            batch_size = p1.shape[0]
            dist_min = torch.ones(batch_size) * self.bar_thc * 3

            delta_pos = p1 - p2
            dist = torch.sqrt(torch.sum((delta_pos ** 2), dim=-1))

            return dist < dist_min

        def _get_state(arm_info, point_info, center_coord):
            # return the distance (dx, dy) between arm finger point with blue point
            return torch.cat([arm_info[:, 0].view(-1, 1), arm_info[:, 3].view(-1, 1),
                              (arm_info[:, 1:3] - point_info) / 200,
                              (arm_info[:, 4:6] - point_info) / 200,
                              (arm_info[:, 4:6] - center_coord) / 200], dim=-1)

        batch_size = u.shape[0]
        center_coord = torch.from_numpy(self.center_coord).unsqueeze(0).repeat(batch_size, 1)
        center_coord = center_coord.type(x.type())

        # action = (node1 angular v, node2 angular v)
        u = torch.clamp(u, -self.action_bound[0], self.action_bound[1])

        arm1rad = x[:, 0]
        arm2rad = x[:, 3]

        point_info = x[:, 4:6] + torch.cat([self.arm2l * torch.cos(arm2rad.view(-1, 1)), self.arm2l * torch.sin(arm2rad.view(-1, 1))], dim=-1)

        arm1rad = arm1rad + u[:, 0] * self.dt
        arm2rad = arm2rad + u[:, 1] * self.dt

        arm1rad %= np.pi * 2
        arm2rad %= np.pi * 2

        arm1rad = arm1rad.view(-1, 1)
        arm2rad = arm2rad.view(-1, 1)

        arm1dx_dy = torch.cat([self.arm1l * torch.cos(arm1rad), self.arm1l * torch.sin(arm1rad)], dim=-1)
        arm2dx_dy = torch.cat([self.arm2l * torch.cos(arm2rad), self.arm2l * torch.sin(arm2rad)], dim=-1)
        arm1xy = center_coord + arm1dx_dy  # (x1, y1)
        arm2xy = arm1xy + arm2dx_dy  # (x2, y2)

        in_col = is_collision(arm2xy, center_coord)
        new_arm = torch.cat([arm1rad, arm1xy[:, 0:2], arm2rad, arm2xy[:, 0:2]], dim=-1)
        x_new = _get_state(new_arm, point_info, center_coord)
        x_new[in_col, :] = x[in_col, :]

        return x_new

    def _reset_arm(self):
        arm1rad, arm2rad = np.random.rand(2) * np.pi * 2
        self.arm_info[0] = arm1rad
        self.arm_info[3] = arm2rad
        arm1dx_dy = np.array([self.arm1l * np.cos(arm1rad), self.arm1l * np.sin(arm1rad)])
        arm2dx_dy = np.array([self.arm2l * np.cos(arm2rad), self.arm2l * np.sin(arm2rad)])
        self.arm_info[1:3] = self.center_coord + arm1dx_dy  # (x1, y1)
        self.arm_info[4:6] = self.arm_info[1:3] + arm2dx_dy  # (x2, y2)

    def reset(self):
        self.get_point = False
        self.grab_counter = 0

        if self.mode == 'hard':
            pxy = np.clip(np.random.rand(2) * self.viewer_xy[0], 100, 300)
            self.point_info[:] = pxy
        else:
            self.point_info[:] = self.point_info_init

        self._reset_arm()
        while self.is_collision(self.arm_info[-2:], self.center_coord):
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
        arm_end = np.vstack([self.arm_info[1:3], self.arm_info[4:6]])
        t_arms = np.ravel(arm_end - self.point_info)
        return np.hstack([self.arm_info[0], self.arm_info[3],
                          (t_arms[0:2] - self.point_info) / 200,
                          (t_arms[2:4] - self.point_info) / 200,
                          (self.arm_info[4:6] - self.center_coord) / 200]), t_arms[-2:]

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

        self.label = pyglet.text.Label("test", color=(0, 0, 0, 255), font_size=10, x=width//2, y=10,
                                       anchor_x='center', anchor_y='center')

    def render(self):
        self.label.text = f'{((self.arm_info[3] / np.pi) * 180):.2f}'
        self._update_arm()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()

    def on_draw(self):
        self.clear()
        self.batch.draw()
        self.label.draw()
        # self.fps_display.draw()

    def _update_arm(self):
        point_l = self.point_l
        point_box = (self.point_info[0] - point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] - point_l,
                     self.point_info[0] + point_l, self.point_info[1] + point_l,
                     self.point_info[0] - point_l, self.point_info[1] + point_l)
        self.point.vertices = point_box

        arm1_coord = (*self.center_coord, *(self.arm_info[1:3]))  # (x0, y0, x1, y1)
        arm2_coord = (*(self.arm_info[1:3]), *(self.arm_info[4:6]))  # (x1, y1, x2, y2)
        arm1_thick_rad = np.pi / 2 - self.arm_info[0]
        x01, y01 = arm1_coord[0:2] + np.array([-np.cos(arm1_thick_rad), np.sin(arm1_thick_rad)]) * self.bar_thc
        x02, y02 = arm1_coord[0:2] + np.array([np.cos(arm1_thick_rad), -np.sin(arm1_thick_rad)]) * self.bar_thc
        x11, y11 = arm1_coord[2:4] + np.array([np.cos(arm1_thick_rad), -np.sin(arm1_thick_rad)]) * self.bar_thc
        x12, y12 = arm1_coord[2:4] + np.array([-np.cos(arm1_thick_rad), np.sin(arm1_thick_rad)]) * self.bar_thc
        arm1_box = (x01, y01, x02, y02, x11, y11, x12, y12)
        arm2_thick_rad = np.pi / 2 - self.arm_info[7]
        x11_, y11_ = arm2_coord[0:2] + np.array([-np.cos(arm2_thick_rad), np.sin(arm2_thick_rad)]) * self.bar_thc
        x12_, y12_ = arm2_coord[0:2] + np.array([np.cos(arm2_thick_rad), -np.sin(arm2_thick_rad)]) * self.bar_thc
        x21, y21 = arm2_coord[2:4] + np.array([np.cos(arm2_thick_rad), -np.sin(arm2_thick_rad)]) * self.bar_thc
        x22, y22 = arm2_coord[2:4] + np.array([-np.cos(arm2_thick_rad), np.sin(arm2_thick_rad)]) * self.bar_thc
        arm2_box = (x11_, y11_, x12_, y12_, x21, y21, x22, y22)
        self.arm1.vertices = arm1_box
        self.arm2.vertices = arm2_box

    def on_key_press(self, symbol, modifiers):
        if symbol == pyglet.window.key.UP:
            self.arm_info[0] += .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.DOWN:
            self.arm_info[0] -= .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.LEFT:
            self.arm_info[3] += .1
            print(self.arm_info[:, 1:3] - self.point_info)
        elif symbol == pyglet.window.key.RIGHT:
            self.arm_info[3] -= .1
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