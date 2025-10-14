import pybullet as p  
import time   

physicsClient = p.connect(p.GUI)

p.loadSDF("box.sdf")

duration = 100

for i in range(duration):
    p.stepSimulation()
    time.sleep(1/60)

p.disconnect() 