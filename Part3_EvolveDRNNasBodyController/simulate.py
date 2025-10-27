import eas
import ctrnn

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

pyrosim.Prepare_To_Simulate(robotId)

duration = 5000
t = np.linspace(0,1,num=duration)

nnsize = 10
motor_outputs = 2

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0

def fitnessFunction(genotype):

    nn = ctrnn.CTRNN(nnsize,0,motor_outputs)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))


    output = np.zeros((duration,nnsize))
    motorout = np.zeros((duration,2))

    linkState = p.getLinkState(robotId,0)
    posx_start = linkState[0][0]
    posy_start = linkState[0][1]

    for i in range(duration):
        nn.step(dt,[])

    for i in range(duration):
        nn.step(dt,[])
        output[i] = nn.Output
        motorout[i] = nn.out()
        motoroutput = nn.out()
        p.stepSimulation()

        pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                    jointName="0_1", 
                                    controlMode = p.POSITION_CONTROL,
                                    targetPosition = (motoroutput[0]*2-1)*np.pi/4,
                                    maxForce = 500
                                    )
        
        pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                    jointName="1_2", 
                                    controlMode = p.POSITION_CONTROL,
                                    targetPosition =  (motoroutput[1]*2-1)*np.pi/4,
                                    maxForce = 500
                                    )    
        
        time.sleep(1/1000) # NEW 

    linkState = p.getLinkState(robotId,0)
    posx_end = linkState[0][0]
    posy_end = linkState[0][1]

    distance = np.sqrt((posx_start - posx_end)**2 + (posy_start - posy_end)**2)
    print(distance)

    return distance, output, motorout

# Load
best = np.load("bestgenotype.npy")
fit, output, motorout = fitnessFunction(best)
print(fit)

p.disconnect() 

# Plot activity
plt.plot(output,alpha=0.5)
plt.plot(motorout.T[0],'k')
plt.plot(motorout.T[1],'k:')
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()

