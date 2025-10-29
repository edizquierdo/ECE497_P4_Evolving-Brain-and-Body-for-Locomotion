import pyrosim.pyrosim as pyrosim

def World(n,xnum,ynum):
    pyrosim.Start_SDF("boxes.sdf")
    for x in range(xnum):
        for y in range(ynum):
            for i in range(n):
                s = 1 - i/n
                pyrosim.Send_Cube(name="Box", pos=[x,y,0.5 + i] , size=[s,s,s])
    pyrosim.End()

def Robot():
    pyrosim.Start_URDF("robot.urdf")
    pyrosim.Send_Cube(name="0", pos=[0,0,1.5] , size=[1,1,1]) ## Cannot 
    pyrosim.Send_Joint(name="0_1", parent="0", child="1", type = "revolute", position=[0.5,0,2])
    pyrosim.Send_Cube(name="1", pos=[0.5,0,0.5] , size=[1,1,1])
    pyrosim.Send_Joint(name="1_2", parent="1", child="2", type = "revolute", position=[1,0,0])
    pyrosim.Send_Cube(name="2", pos=[0.5,0,-0.5] , size=[1,1,1])
    pyrosim.End()

Robot()

#World(10,5,5)
