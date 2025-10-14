This is the starter code for Project #4. Here are the instructions. 

First, install PyBullet (Python-based physics engine).
- Windows users need to install Visual Studio 2022 to work with C++. If you do not already have it, download and install: https://visualstudio.microsoft.com/downloads/
- Windows users also need to install or update wheel. Open a Command Prompt in Windows and type: pip3 install wheel
- Open a Command Prompt in Windows or Terminal on a Mac and type: pip3 install pybullet
- If you don't have pip3 or are not familiar with it, get it here: https://pypi.org/project/pip/
- If you have pip installed but pip3 is not recognized, try: py -m pip install pybullet

Second, download the code in this GitHub repository, which includes: 
- Two starter code files: generate.py and simulate.py
- Pyrosim package, which will allow us to more easily send information to pybullet, and get information back from it. Pyrosim stands for a [Py]thon [Ro]bot [Sim]ulator.
- Make sure you place both the Pyrosim package and the two files in the same directory.

Third, make sure everything runs: 
- First run python generate.py
- Then run python simulate.py

This should launch your Physics simulation: 
- You can look around your world by rotating the virtual camera by clicking and dragging with the mouse. You can also hold the left control button down and move with the mouse (Mac users: hold down CTRL, click and drag with the mouse or press and drag on a trackpad).
- A two-fingered swipe on a trackpad, or rolling a mouse's scroll wheel, will move the position of the virtual camera.
- You can also manipulate the objects in the world. 
- You can "grab" the cube object that you create with your mouse and "throw" them by moving your mouse. Hold down SHIFT, click on the cube, and drag the mouse. You should see the cube, but not the virtual camera (Mac users: Click on the cube and drag the mouse).

  
