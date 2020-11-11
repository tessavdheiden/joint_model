from matplotlib import pyplot as plt
import numpy as np
import torch


def ppo_heatmap():
    from policy import Net

    x_pxl, y_pxl = 300, 400

    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    net = Net()
    net.load_state_dict(torch.load('net_params.pkl'))
    value_map = net(state)[1].view(y_pxl, x_pxl).detach().numpy()

    fig = plt.figure(figsize=(20,5))
    fig.suptitle('PPO')
    ax = fig.add_subplot(141)
    im = ax.imshow(value_map, cmap=plt.cm.spring, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Value Map')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    action_map = net(state)[0][0][:, 0].view(y_pxl, x_pxl).detach().numpy()
    ax = fig.add_subplot(142)
    im = ax.imshow(action_map, cmap=plt.cm.winter, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Action P Map')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    action_map = net(state)[0][0][:, 1].view(y_pxl, x_pxl).detach().numpy()
    ax = fig.add_subplot(143)
    im = ax.imshow(action_map, cmap=plt.cm.winter, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Action I Map')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    action_map = net(state)[0][0][:, 2].view(y_pxl, x_pxl).detach().numpy()
    ax = fig.add_subplot(144)
    im = ax.imshow(action_map, cmap=plt.cm.winter, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Action D Map')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])

    plt.tight_layout()
    plt.savefig('img/ppo_heatmap.png')
    plt.show()


ppo_heatmap()
