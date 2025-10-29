import eas
import ctrnn

import pybullet as p  
import pyrosim.pyrosim as pyrosim
import pybullet_data
import time   

import numpy as np
import matplotlib.pyplot as plt

# physicsClient = p.connect(p.GUI) 
physicsClient = p.connect(p.DIRECT) # NEW
p.setAdditionalSearchPath(pybullet_data.getDataPath())
p.configureDebugVisualizer(p.COV_ENABLE_GUI,0)
p.setGravity(0,0,-9.8)

planeId = p.loadURDF("plane.urdf")
robotId = p.loadURDF("robot.urdf")

pyrosim.Prepare_To_Simulate(robotId)

transient = 1000
duration = 5000                 

nnsize = 10
motor_outputs = 2

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0

def reset_robot(robotId, base_pos=[0,0,1], base_orn=[0,0,0,1]):
    # Reset base position and orientation
    p.resetBasePositionAndOrientation(robotId, base_pos, base_orn)

    # zero out velocity
    p.resetBaseVelocity(robotId, [0,0,0], [0,0,0])

    # Reset all joint angles and velocities
    num_joints = p.getNumJoints(robotId)
    for j in range(num_joints):
        p.resetJointState(robotId, j, targetValue=0.0, targetVelocity=0.0)

def fitnessFunction(genotype):
    # Reset joints / Reset the coordinates the body 
    reset_robot(robotId)

    nn = ctrnn.CTRNN(nnsize,0,motor_outputs)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))

    # Simulate both NN and Body for a little while 
    # without connecting them, so that transients pass 
    # for both of them
    for i in range(transient):
        nn.step(dt,[])
        p.stepSimulation()

    # Test period
    # Get starting position (after the transient)
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
generations = 100

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# Get best evolved network
af,bf,bi = ga.fitStats()

# Save 
np.save("bestgenotype.npy",bi)

p.disconnect() 

