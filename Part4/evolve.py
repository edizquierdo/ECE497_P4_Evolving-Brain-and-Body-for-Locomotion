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
duration = 4000                 

nnsize = 5
sensor_inputs = 1
motor_outputs = 2

dt = 0.01
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0

nn = ctrnn.CTRNN(nnsize,sensor_inputs,motor_outputs)

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

    # Set and initialize the neural network
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))

    # Simulate both NN and Body for a little while 
    # without connecting them, so that transients pass 
    # for both of them
    for i in range(transient):
        nn.step(dt,[0.0])
        p.stepSimulation()

    # Test period
    # Get starting position (after the transient)
    linkState = p.getLinkState(robotId,0)
    posx_start = linkState[0][0]
    posy_start = linkState[0][1]
    posx_current = linkState[0][0]
    posy_current = linkState[0][1]
    posz_current = linkState[0][2]   

    distance_traveled = 0.0 # distance traveled in the x-y plane at each step, to be maximized
    distance_jumped = 0.0 # amount of movement up and down, to be minimized

    # Simulate both NN and Body
    for i in range(duration):
        legsensor = pyrosim.Get_Touch_Sensor_Value_For_Link("2")
        for ttt in range(10):
            nn.step(dt,[legsensor])
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
        posx_past = posx_current
        posy_past = posy_current
        posz_past = posz_current   
        linkState = p.getLinkState(robotId,0)
        posx_current = linkState[0][0]
        posy_current = linkState[0][1]
        posz_current = linkState[0][2]    
        distance_traveled += np.sqrt((posx_current - posx_past)**2 + (posy_current - posy_past)**2)
        distance_jumped += np.sqrt((posz_current - posz_past)**2)

    # Get final position 
    linkState = p.getLinkState(robotId,0)
    posx_end = linkState[0][0]
    posy_end = linkState[0][1]

    distance_final = np.sqrt((posx_start - posx_end)**2 + (posy_start - posy_end)**2)

    return (10 * distance_final) + distance_traveled - distance_jumped

# EA Params
popsize = 10
genesize = nnsize*nnsize + 2*nnsize + sensor_inputs*nnsize + motor_outputs*nnsize 
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

