import pybullet as p    #1.1 
import pybullet_data
import pyrosim.pyrosim as ps #2.1
import numpy as np
import time 

physicsClient = p.connect(p.GUI) #1.2

p.configureDebugVisualizer(p.COV_ENABLE_GUI,0) #1.5

p.setAdditionalSearchPath(pybullet_data.getDataPath())

duration = 10000

p.setGravity(0,0,-9.8)
planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("body.urdf")
p.loadSDF("box.sdf")  #2.X

ps.Prepare_To_Simulate(robotId)

x = np.linspace(0,10*np.pi,duration)
signal = np.sin(x)

for i in range(duration):
    ps.Set_Motor_For_Joint(bodyIndex=robotId,jointName=b'Foot_Torso',controlMode=p.POSITION_CONTROL,targetPosition=signal[i],maxForce=500)
    p.stepSimulation() #1.4
    time.sleep(1/500)

p.disconnect() #1.3