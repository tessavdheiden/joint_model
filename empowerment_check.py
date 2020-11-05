import torch
import numpy as np
import matplotlib.pyplot as plt


def visualize_predictions(empowerment):
    x_pxl, y_pxl = 300, 400

    state = torch.Tensor([[np.cos(theta), np.sin(theta), thetadot]
                          for thetadot in np.linspace(-8, 8, y_pxl)
                          for theta in np.linspace(-np.pi, np.pi, x_pxl)])

    v = empowerment.compute(state)
    value_map = v.view(y_pxl, x_pxl).detach().numpy()

    fig = plt.figure()
    ax = fig.add_subplot(111)
    im = ax.imshow(value_map, cmap=plt.cm.spring, interpolation='bicubic')
    plt.colorbar(im, shrink=0.5)
    ax.set_title('Empowerment Landscape')
    ax.set_xlabel('$\\theta$')
    ax.set_xticks(np.linspace(0, x_pxl, 5))
    ax.set_xticklabels(['$-\\pi$', '$-\\pi/2$', '$0$', '$\\pi/2$', '$\\pi$'])
    ax.set_ylabel('$\\dot{\\theta}$')
    ax.set_yticks(np.linspace(0, y_pxl, 5))
    ax.set_yticklabels(['-8', '-4', '0', '4', '8'])
    plt.savefig('img/empowerment_landscape.png')
