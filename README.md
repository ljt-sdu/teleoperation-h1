# Introduction
This repository is modified from [unitree avp_teleoperate h1](https://github.com/unitreerobotics/avp_teleoperate/tree/h1). It implements teleoperation of the humanoid robot Unitree H1(arms only) using Apple Vision Pro.



# Prerequisites

We tested our code on Ubuntu 20.04, other operating systems may be configured differently.  

## Installation
Please refer to the following link for installation：
1) [Unitree avp_teleoperate h1](https://github.com/unitreerobotics/avp_teleoperate/tree/h1)
2) [Official documentation](https://support.unitree.com/home/en/Teleoperation)
3) [OpenTeleVision](https://github.com/OpenTeleVision/TeleVision)


## Image Server

Copy `image_server.py` in the `avp_teleoperate/teleop/image_server` directory to the PC of Unitree H1, and execute the following command **in the PC**:

```bash
sudo python image_server.py
```

After image service is started, you can use `image_client.py` **in the Host(192.168.123.162)** terminal to test whether the communication is successful:

```bash
python image_client.py
```

## Start

> Warning : All persons must maintain an adequate safety distance from the robot to avoid danger!

```bash
cd teleop
python teleop_arm.py
```

# Acknowledgement

This code builds upon following open-source code-bases. Please visit the URLs to see the respective LICENSES:
1) https://github.com/unitreerobotics/avp_teleoperate
2) https://github.com/OpenTeleVision/TeleVision
3) https://github.com/dexsuite/dex-retargeting
4) https://github.com/vuer-ai/vuer
5) https://github.com/stack-of-tasks/pinocchio
6) https://github.com/casadi/casadi
7) https://github.com/meshcat-dev/meshcat-python
8) https://github.com/zeromq/pyzmq
9) https://github.com/unitreerobotics/unitree_dds_wrapper
10) https://github.com/tonyzhaozh/act
11) https://github.com/facebookresearch/detr
