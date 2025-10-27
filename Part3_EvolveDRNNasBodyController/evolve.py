import eas
import ctrnn

import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time   

import numpy as np
import matplotlib.pyplot as plt

#physicsClient = p.connect(p.GUI)
physicsClient = p.connect(p.DIRECT) # NEW
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")

pyrosim.Prepare_To_Simulate(robotId)

transient = 1000
duration = 4000                 

nnsize = 10
motor_outputs = 2

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0

def fitnessFunction(genotype):
    # Reset the body
    # Reset the position back to 0s
    # Reset the angles back to some standard

    nn = ctrnn.CTRNN(nnsize,0,motor_outputs)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))

    # Test period
    # Get starting position
    linkState = p.getLinkState(robotId,0)
    posx_start = linkState[0][0]
    posy_start = linkState[0][1]

    # Simulate both NN and Body
    for i in range(duration):
        nn.step(dt,[])
        motoroutput = nn.out()
        p.stepSimulation()

        pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                    jointName="0_1", 
                                    controlMode = p.POSITION_CONTROL,
                                    targetPosition = (motoroutput[0]*2-1)*np.pi/4, # NEW
                                    maxForce = 500
                                    )
        
        pyrosim.Set_Motor_For_Joint(bodyIndex= robotId, 
                                    jointName="1_2", 
                                    controlMode = p.POSITION_CONTROL,
                                    targetPosition = (motoroutput[1]*2-1)*np.pi/4, # NEW
                                    maxForce = 500 
                                    )    
        
        #time.sleep(1/60) # NEW 

    # Get final position 
    linkState = p.getLinkState(robotId,0)
    posx_end = linkState[0][0]
    posy_end = linkState[0][1]

    distance = np.sqrt((posx_start - posx_end)**2 + (posy_start - posy_end)**2)
    return distance 

# EA Params
popsize = 10
genesize = nnsize*nnsize + 2*nnsize + 2*nnsize # last one is for the output weights 
recombProb = 0.5
mutatProb = 0.01
demeSize = 2
generations = 10

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# Get best evolved network
af,bf,bi = ga.fitStats()

# Save 
np.save("bestgenotype.npy",bi)

p.disconnect() 

