import matplotlib.pyplot as plt
import torch


def visualize_observation_distributions(replay_memory):
    replay_memory.reset_batchptr_train()

    X, U = [], [], []
    for b in range(1):
        batch_dict = replay_memory.next_batch_train()
        x, u = torch.from_numpy(batch_dict["states"]), torch.from_numpy(batch_dict['inputs'])
        x = x.reshape(-1, x.shape[2])
        X.append(x)
        U.append(u.reshape(-1, u.shape[2]))

    X = torch.cat(X, dim=0)

    fig, ax = plt.subplots(ncols=replay_memory.state_dim, nrows=1, figsize=(9, 3))

    for i in range(replay_memory.state_dim):
        ax[i].hist(X[:, 0].numpy(), bins=10)
        ax[i].set_xlabel(f'x at dim={i}')

    plt.tight_layout()
    plt.savefig('img/dist_x.png')
    plt.close()