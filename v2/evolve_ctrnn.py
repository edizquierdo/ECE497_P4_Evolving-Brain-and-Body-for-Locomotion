import eas
import ctrnn
import numpy as np
import matplotlib.pyplot as plt

nnsize = 10

dt = 0.01
duration = 50.0
TimeConstMin = 1.0
TimeConstMax = 2.0
WeightRange = 10.0
BiasRange = 10.0

dur = np.arange(0.0,duration,dt)

def fitnessFunction(genotype):
    nn = ctrnn.CTRNN(nnsize,0,0)
    nn.setParameters(genotype,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
    nn.initializeState(np.zeros(nnsize))
    output = np.zeros((len(dur),nnsize))
    step = 0
    for t in dur:
        nn.step(dt,[])
        output[step] = nn.Output
        step += 1
    SD = (np.sum(np.abs(np.diff(output,axis=0)))/nnsize)/duration
    return(SD)

# EA Params
popsize = 10
genesize = nnsize*nnsize + 2*nnsize
recombProb = 0.5
mutatProb = 0.01
demeSize = 2
generations = 500

# Evolve and visualize fitness over generations
ga = eas.Microbial(fitnessFunction, popsize, genesize, recombProb, mutatProb, demeSize, generations)
ga.run()
ga.showFitness()

# Get best evolved network
af,bf,bi = ga.fitStats()

# show its activity
nn = ctrnn.CTRNN(nnsize,0,0)
nn.setParameters(bi,WeightRange,BiasRange,TimeConstMin,TimeConstMax)
nn.initializeState(np.zeros(nnsize))
output = np.zeros((len(dur),nnsize))
k = 0
for t in dur:
    nn.step(dt,[])
    output[k] = nn.Output
    k += 1

# Plot activity
plt.plot(dur,output)
plt.xlabel("Time")
plt.ylabel("Output")
plt.title("Neural activity")
plt.show()

nn.save("ctrnn")