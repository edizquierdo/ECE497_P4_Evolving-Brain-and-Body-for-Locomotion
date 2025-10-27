import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time   
import numpy as np
import matplotlib.pyplot as plt

physicsClient = p.connect(p.GUI)
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")
#p.loadSDF("boxes.sdf")

pyrosim.Prepare_To_Simulate(robotId)

duration = 1000
t = np.linspace(0,1,num=duration)
x = 10
tp1 = np.zeros(duration)
tp2 = np.zeros(duration)

posx = np.zeros(duration)
posy = np.zeros(duration)
posz = np.zeros(duration)

legsensor = np.zeros(duration)

for i in range(duration):
    p.stepSimulation()

    tp1[i] = np.sin(x * t[i]*2*np.pi) * np.pi/4  
    tp2[i] = np.cos(x * t[i]*2*np.pi) * np.pi/4

    pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                jointName="0_1", 
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = tp1[i],
                                maxForce = 500
                                )
    pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                jointName="1_2", 
                                controlMode = p.POSITION_CONTROL,
                                targetPosition = tp2[i],
                                maxForce = 500
                                )    
    
    linkState = p.getLinkState(robotId,0)
    posx[i] = linkState[0][0]
    posy[i] = linkState[0][1]
    posz[i] = linkState[0][2]

    #legsensor[i] = pyrosim.Get_Touch_Sensor_Value_For_Link("2")

    # tp1[i] = np.sin(x * t[i]*2*np.pi) * np.pi/4  
    # if legsensor[i] > 0:
    #     tp2[i] = np.cos(x * t[i]*2*np.pi) * np.pi/4
    # else:
    #     tp2[i] = 0

    time.sleep(1/60)

p.disconnect() 

plt.plot(tp1,label="Back leg motor")
plt.plot(tp2, label="Front leg motor")
plt.plot(posx, label="x")
plt.plot(posy, label="y")
plt.plot(posz, label="z")
#plt.plot(legsensor, label="FL sensor")
plt.legend()
plt.xlabel("Duration")
plt.ylabel("Target")
plt.show()