# Jetrat - Scaled autonomous car using Jetson
By [Sudharsan Ananth](https://sudharsanananth.wixsite.com/sudharsan) 

<!-- TABLE OF CONTENTS -->
<details>
  <summary>Table of Contents</summary>
  <ol>
    <li><a href="#introduction">Introduction</a></li>
    <li><a href="#dependencies">Dependencies</a></li>
    <li><a href="#hardware-requirements">Hardware Requirement</a></li>
    <li><a href="#run-the-code">How to run</a></li>
    <li><a href="#license">License</a></li>
  </ol>
</details>


## Introduction 

This is a repositorie containing the code for Jetrat which is a self-driving car, based on [Nvidia's model architecture](https://images.nvidia.com/content/tegra/automotive/images/2016/solutions/pdf/end-to-end-dl-using-px.pdf). This code and the architecture has been updated and improved overtime to be efficient on low powered machines like Jetson. This code has been tested on Jetson Nano 4gb model. Hardware requirement for this is as same the JetRacer. The controller module has been updated to suppert more modern controllers such as xBox, PS3, PS4 etc. List of all the controllers can be found `Controller.py` and `Gamepad.py`. Simply change the controller type inside this file to support your controller. This project is made possible by the open source support and codes below. Huge thanks to all the contributers. 

Training of the model can be done using this directory. Training code has been tested in Mac and Windows PC. 


### Short Video of Jetrat (enjoy). 


https://user-images.githubusercontent.com/55453134/218347405-dee79ea0-1b13-40db-bf95-1e4a540c3db0.mov

#### [Watch-full-video-on-youtube](https://youtu.be/gaRUw0A2xp0)


### Reference codes used from sources below. 

* [Jetracer](https://github.com/NVIDIA-AI-IOT/jetracer) 
* [piborg-Gamepad](https://github.com/piborg/Gamepad) 

## Dependencies 

This project is built with the below given major frameworks and libraries. The code is primarily based on python. 

* [Python](https://www.python.org/) 
* [pytorch](https://pytorch.org/)
* [matplotlib](https://matplotlib.org/) 
* [pandas](https://pandas.pydata.org) 
* [openCV](https://docs.opencv.org/4.x/d6/d00/tutorial_py_root.html) 

## Hardware Requirements 

1. [Waveshare-Jetracer-Pro-kit](https://www.waveshare.com/product/jetracer-pro-ai-kit.htm), also refer [Jetracer](https://github.com/NVIDIA-AI-IOT/jetracer) github
2. Generic Game controller 
3. Jetson Nano 4gb
4. imx-219 camera with atleast(180 fov), Only this camera type supported by Jetson. 


## Run the code

Simply clone the repo cd into the right directory and run 'main.py' using the below commands. Step-by-Step instructions given below. 

1. Clone the repository using 
   ```sh
   git clone https://github.com/sudharsan-007/jetrat.git
   ```

2. cd into the directory RL-DQN-Snake-Game
   ```sh
   cd jetrat
   ```

3. Install requirements/packages (skip and try the next step)
   ```sh 
   pip install -r packages.txt
   ```

4. Ideally everything should be preinstalled and only `requirements.txt` should be installed. 
   ```sh
   pip install -r requirements.txt
   ```

5. Edit the controller code and test with `Racestick.py`, List of all the supported controller is given in `Gamepad.py`. 
    ```sh 
    conda install pytorch torchvision torchaudio cpuonly -c pytorch
    ```

6. Modify and test all the button inputs of the controller. 

7. Run `main.py` and collect a small data for camera calibration. Use Checkerboard and place them at different locations. 
    ```sh 
    python main.py
    ```

8. Run `camera_calibration.py` to generate calibration matrix. 
   ```sh
   python camera_calibration.py`
   ```

9. Run `main.py` collect data, train, drive, or set it to autonomous mode. Explore and enjoy the new possiblities. 
    ```sh 
    python main.py
    ```

<!-- LICENSE -->
## License

Distributed under the GNU General Public License v3.0. See `LICENSE.txt` for more information.

<p align="right">(<a href="#top">back to top</a>)</p>
