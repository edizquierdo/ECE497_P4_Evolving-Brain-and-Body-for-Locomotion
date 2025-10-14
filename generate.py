import pyrosim.pyrosim as ps #2.1

l = 1 # length
w = 1 # width 
h = 1 # height

x = 0
y = 0 
z = 0.0

def Create_World():
    ps.Start_SDF("box.sdf") #2.2
    for i in range(10):
        ps.Send_Cube(name="Box", pos=[x,y,z], size=[l,w,h]) #2.3
        z += l
        l = 0.6*l
        w = 0.9*w
        h = 0.9*h
    ps.End() #2.4

def Create_Robot():
    ps.Start_URDF("body.urdf")
    ps.Send_Cube(name="Foot",pos=[0,0,0.5],size=[l,w,h])
    ps.Send_Joint(name="Foot_Torso",parent="Foot",child="Torso",type="slider",position=[0.5,0.0,1.0])
    ps.Send_Cube(name="Torso",pos=[0.5,0,0.5],size=[l,w,h])
    ps.End()

Create_World()
Create_Robot()