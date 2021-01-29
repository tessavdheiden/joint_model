# joint_model

## Overview

This repository contains various systems for which the empowerment landscape can be computed. 

```
python3 train/empowerment_train.py --env 0
```

A list of the sytems:
- 0: Pendulum
- 1: Ball in box
- 2: Robot arm

Empowerment can be computed with and without a filter that estimates hidden states. 

```
python3 train/empowerment_train.py --use_filter 1
```

Finally, a policy maximizing empowerment can be trained.

```
python3 train/policy_train.py
```