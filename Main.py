import Model3Prod

import time
import math
import random
from MPVRPIR import *
import Constructive

cp_off = {}
cp_off['CutProposition1'] = {'active':False, 'e':1e-6}
cp_off['CutProposition2'] = {'active':False, 'e':1e-6}
cp_off['CutProposition3'] = {'active':False, 'e':1e-6}
cp_off['CutProposition4'] = {'active':False, 'e':1e-6}
cp_off['CutProposition5'] = {'active':False, 'e':1e-6}
cp_off['CutProposition6'] = {'active':False, 'e':1e-6}
cp_off['CutProposition7'] = {'active':False, 'e':1e-6}
cp_off['CutProposition8'] = {'active':False, 'e':1e-6}
cp_off['CutProposition9a'] = {'active':False, 'e':1e-6}
cp_off['CutProposition9b'] = {'active':False, 'e':1e-6}
cp_off['CutProposition10'] = {'active':False, 'e':1e-6}
cp_off['CutProposition11a'] = {'active':False, 'e':1e-6}
cp_off['CutProposition11b'] = {'active':False, 'e':1e-6}
cp_off['CutProposition12a'] = {'active':False, 'e':1e-6}
cp_off['CutProposition12b'] = {'active':False, 'e':1e-6}
cp_off['CutProposition13'] = {'active':False, 'e':1e-6}

cp_forward_short = {}
cp_forward_short['CutProposition1'] = {'active':True, 'e':1e-6}
cp_forward_short['CutProposition2'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition3'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition4'] = {'active':True, 'e':0.5}
cp_forward_short['CutProposition5'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition6'] = {'active':True, 'e':1e-6}
cp_forward_short['CutProposition7'] = {'active':True, 'e':0.5}
cp_forward_short['CutProposition8'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition9a'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition9b'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition10'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition11a'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition11b'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition12a'] = {'active':True, 'e':0.1}
cp_forward_short['CutProposition12b'] = {'active':False, 'e':1e-6}
cp_forward_short['CutProposition13'] = {'active':False, 'e':1e-6}

cp_backward_short = {}
cp_backward_short['CutProposition1'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition2'] = {'active':True, 'e':0.1}
cp_backward_short['CutProposition3'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition4'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition5'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition6'] = {'active':True, 'e':0.9}
cp_backward_short['CutProposition7'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition8'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition9a'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition9b'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition10'] = {'active':True, 'e':0.1}
cp_backward_short['CutProposition11a'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition11b'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition12a'] = {'active':False, 'e':1e-6}
cp_backward_short['CutProposition12b'] = {'active':True, 'e':1e-6}
cp_backward_short['CutProposition13'] = {'active':True, 'e':0.5}

cp_forward_long = {}
cp_forward_long['CutProposition1'] = {'active':True, 'e':0.1}
cp_forward_long['CutProposition2'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition3'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition4'] = {'active':True, 'e':0.5}
cp_forward_long['CutProposition5'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition6'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition7'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition8'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition9a'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition9b'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition10'] = {'active':True, 'e':0.1}
cp_forward_long['CutProposition11a'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition11b'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition12a'] = {'active':True, 'e':0.5}
cp_forward_long['CutProposition12b'] = {'active':False, 'e':1e-6}
cp_forward_long['CutProposition13'] = {'active':False, 'e':1e-6}

cp_backward_long = {}
cp_backward_long['CutProposition1'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition2'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition3'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition4'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition5'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition6'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition7'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition8'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition9a'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition9b'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition10'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition11a'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition11b'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition12a'] = {'active':True, 'e':1e-6}
cp_backward_long['CutProposition12b'] = {'active':False, 'e':1e-6}
cp_backward_long['CutProposition13'] = {'active':False, 'e':1e-6}

cut_parameters = {'short':[cp_off, cp_forward_short, cp_backward_short], 'long':[cp_off, cp_forward_long, cp_backward_long]}

def SolveProblem(problem, constructive = True, Model = False, name = 'unnamed', prin=True, cutParameters = None):
    if constructive:
        ts = time.time()
        sol = Constructive.Constructive(problem)
        if sol != None:
            solheu, tt, costheu = sol
            dheu = len(solheu)
        else:
            solheu, tt, costheu, dheu = None, None, -1, -1
        te = time.time()
        theu = te-ts
        
        if prin:
            print('Constructive heuristic')
            if sol == None:
                print('  No solution found')
            else:
                print('  Solution cost:', costheu, dheu, 'days')
            print('  Time:', te-ts)

    if Model:
        info, _ = Model3Prod.Solve(problem, (solheu, tt, costheu) if sol != None else None, integer=False)
        lbroot = info['lb']
        
        ts = time.time()
        info, sol = Model3Prod.Solve(problem, (solheu, tt, costheu) if sol != None else None, integer=True, cutParameters = cutParameters)
        te = time.time()

        if prin:
            print('Model')
            if info['status'] == 'infeasible':
                print('  INFEASIBLE')
            else:
                print('  ub bb', info['ub'], len(sol), 'days')
                print('  lb bb', info['lb'])
                print('  lin rel', lbroot)
                print('  lb root (cuts)', info['lbroot'])
                print('  num nodes', info['numnodes'])
                print('  time', te-ts)

def GetProblem(type, numVertices):
    numTeams = 2
    numTasks = 4
    
    if type == 'short':
        baseTimes = [1,2,3,4,5]
        maxDays = int(max(4,numVertices/2))
    else:
        baseTimes = [6,7,8,9,10]
        maxDays = int(max(4,numVertices))

    if random.random() > 0.5:
        teamProd = [1,2]
        teamCost = [1,2.25]
        maxTeams = [3,2]
    else:
        teamProd = [1,2]
        teamCost = [1.25, 2]
        maxTeams = [2,3]

    instanceArgs = {
        'numVertices':numVertices, 
        'numTasks':numTasks, 
        'numTeams':numTeams, 
        'baseTimes':baseTimes,
        'teamProd':teamProd,
        'teamCost':teamCost,
        'maxTeamsAll': sum(maxTeams), 
        'maxTeams':maxTeams,
        'maxDays':maxDays}

    return instanceArgs 

def OneInstanceRandom(type, n, i):
        print('Seed', i)
        random.seed(i)

        instanceArgs = GetProblem(type, n)
        print('Instance parameters', instanceArgs)

        instance = MPVRPIR(**instanceArgs) 

        SolveProblem(problem = instance, constructive = True, Model = True, cutParameters=cp_backward_short if type == 'short' else cp_backward_long)

def OneInstanceFile(filename):
        print(filename)
        instance = MPVRPIR()
        instance.ReadInstance(filename)
        SolveProblem(problem = instance, constructive = True, Model = True, cutParameters=cp_backward_short if type == 'short' else cp_backward_long)

#### Uncomment to execute a single random instance
# format type (short or long) n seed
'''
import sys
OneInstanceRandom(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
'''

#### Uncomment to execute a single instance from a file
import sys
OneInstanceFile(sys.argv[1])

#### Uncomment to execute instances in the folder
'''
from os import listdir
from os.path import isfile, join

instancefolder = './Instances'
for f in sorted(listdir(instancefolder)):
    i = join(instancefolder, f)
    if isfile(i):
        instance = MPVRPIR()
        instance.ReadInstance(i)

        print(i)
        #SolveProblem(problem = instance, constructive = True, Model = True, cutParameters = cp_forward_short, prin=True)
        SolveProblem(problem = instance, constructive = True, Model = False)
        print('\n\n\n\n')
'''
