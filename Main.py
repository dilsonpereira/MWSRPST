import Model3Prod

import time
import math
import random
from MPVRPIR import *
import Constructive

cp_off = {}
cp_off['cut1'] = {'active':False, 'e':1e-6}
cp_off['cut2'] = {'active':False, 'e':1e-6}
cp_off['cut3a'] = {'active':False, 'e':1e-6}
cp_off['cut3b'] = {'active':False, 'e':1e-6}
cp_off['cut4a'] = {'active':False, 'e':1e-6}
cp_off['cut5'] = {'active':False, 'e':1e-6}
cp_off['cut7'] = {'active':False, 'e':1e-6}
cp_off['cut8a'] = {'active':False, 'e':1e-6}
cp_off['cut8b'] = {'active':False, 'e':1e-6}
cp_off['cut9b'] = {'active':False, 'e':1e-6}
cp_off['cut10'] = {'active':False, 'e':1e-6}
cp_off['cut11'] = {'active':False, 'e':1e-6}
cp_off['cut11b'] = {'active':False, 'e':1e-6}
cp_off['cut11c'] = {'active':False, 'e':1e-6}
cp_off['cut11d'] = {'active':False, 'e':1e-6}
cp_off['cutSets'] = {'active':False, 'e':1e-6}

cp_forward_short = {}
cp_forward_short['cut1'] = {'active':False, 'e':1e-6}
cp_forward_short['cut2'] = {'active':False, 'e':1e-6}
cp_forward_short['cut3a'] = {'active':True, 'e':0.5}
cp_forward_short['cut3b'] = {'active':False, 'e':1e-6}
cp_forward_short['cut4a'] = {'active':True, 'e':1e-6}
cp_forward_short['cut5'] = {'active':False, 'e':1e-6}
cp_forward_short['cut7'] = {'active':False, 'e':1e-6}
cp_forward_short['cut8a'] = {'active':False, 'e':1e-6}
cp_forward_short['cut8b'] = {'active':False, 'e':1e-6}
cp_forward_short['cut9b'] = {'active':True, 'e':0.5}
cp_forward_short['cut10'] = {'active':False, 'e':1e-6}
cp_forward_short['cut11'] = {'active':False, 'e':1e-6}
cp_forward_short['cut11b'] = {'active':False, 'e':1e-6}
cp_forward_short['cut11c'] = {'active':True, 'e':0.1}
cp_forward_short['cut11d'] = {'active':False, 'e':1e-6}
cp_forward_short['cutSets'] = {'active':True, 'e':1e-6}

cp_backward_short = {}
cp_backward_short['cut1'] = {'active':True, 'e':0.1}
cp_backward_short['cut2'] = {'active':True, 'e':1e-6}
cp_backward_short['cut3a'] = {'active':True, 'e':1e-6}
cp_backward_short['cut3b'] = {'active':False, 'e':1e-6}
cp_backward_short['cut4a'] = {'active':True, 'e':0.9}
cp_backward_short['cut5'] = {'active':True, 'e':0.5}
cp_backward_short['cut7'] = {'active':True, 'e':1e-6}
cp_backward_short['cut8a'] = {'active':False, 'e':1e-6}
cp_backward_short['cut8b'] = {'active':False, 'e':1e-6}
cp_backward_short['cut9b'] = {'active':True, 'e':1e-6}
cp_backward_short['cut10'] = {'active':True, 'e':0.1}
cp_backward_short['cut11'] = {'active':False, 'e':1e-6}
cp_backward_short['cut11b'] = {'active':False, 'e':1e-6}
cp_backward_short['cut11c'] = {'active':False, 'e':1e-6}
cp_backward_short['cut11d'] = {'active':True, 'e':1e-6}
cp_backward_short['cutSets'] = {'active':True, 'e':1e-6}

cp_forward_long = {}
cp_forward_long['cut1'] = {'active':False, 'e':1e-6}
cp_forward_long['cut2'] = {'active':False, 'e':1e-6}
cp_forward_long['cut3a'] = {'active':True, 'e':0.5}
cp_forward_long['cut3b'] = {'active':False, 'e':1e-6}
cp_forward_long['cut4a'] = {'active':False, 'e':1e-6}
cp_forward_long['cut5'] = {'active':False, 'e':1e-6}
cp_forward_long['cut7'] = {'active':False, 'e':1e-6}
cp_forward_long['cut8a'] = {'active':False, 'e':1e-6}
cp_forward_long['cut8b'] = {'active':False, 'e':1e-6}
cp_forward_long['cut9b'] = {'active':False, 'e':1e-6}
cp_forward_long['cut10'] = {'active':True, 'e':0.1}
cp_forward_long['cut11'] = {'active':False, 'e':1e-6}
cp_forward_long['cut11b'] = {'active':False, 'e':1e-6}
cp_forward_long['cut11c'] = {'active':True, 'e':0.5}
cp_forward_long['cut11d'] = {'active':False, 'e':1e-6}
cp_forward_long['cutSets'] = {'active':True, 'e':0.1}

cp_backward_long = {}
cp_backward_long['cut1'] = {'active':True, 'e':1e-6}
cp_backward_long['cut2'] = {'active':False, 'e':1e-6}
cp_backward_long['cut3a'] = {'active':True, 'e':1e-6}
cp_backward_long['cut3b'] = {'active':False, 'e':1e-6}
cp_backward_long['cut4a'] = {'active':True, 'e':1e-6}
cp_backward_long['cut5'] = {'active':False, 'e':1e-6}
cp_backward_long['cut7'] = {'active':False, 'e':1e-6}
cp_backward_long['cut8a'] = {'active':False, 'e':1e-6}
cp_backward_long['cut8b'] = {'active':True, 'e':1e-6}
cp_backward_long['cut9b'] = {'active':True, 'e':1e-6}
cp_backward_long['cut10'] = {'active':False, 'e':1e-6}
cp_backward_long['cut11'] = {'active':False, 'e':1e-6}
cp_backward_long['cut11b'] = {'active':False, 'e':1e-6}
cp_backward_long['cut11c'] = {'active':True, 'e':1e-6}
cp_backward_long['cut11d'] = {'active':False, 'e':1e-6}
cp_backward_long['cutSets'] = {'active':True, 'e':1e-6}

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
        
        '''
        print('Constructive heuristic')
        if sol == None:
            print('No solution found')
        else:
            print('  Solution cost:', costheu, len(sol), 'days')
        print('  Time:', te-ts)
        '''

    if Model:
        info, _ = Model3Prod.Solve(problem, (solheu, tt, costheu) if sol != None else None, integer=False)
        lbroot = info['lb']
        
        ts = time.time()
        info, sol = Model3Prod.Solve(problem, (solheu, tt, costheu) if sol != None else None, integer=True, cutParameters = cutParameters)
        te = time.time()

        if info['status'] == 'infeasible':
            if prin:
                print(name, 'INFEASIBLE')

        if prin:
            print('{:>10} {:>8} {:>8} {:>8.2f} {:>8} {:>8.2f} {:>8.2f} {:>8.2f} {:>8} {:>8} {:>8.2f}'.format(name, costheu, dheu, theu, info['ub'], info['lb'], lbroot, info['lbroot'], len(sol), info['numnodes'], te-ts))


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

def ForestProdExperiments():

    import random
    
    seed = list(range(100))
    for n in [3,4]:
        print('Instance Size:', n, '#############################')
        for t in ['short', 'long']:
            print('Type', t, '****************')
            for cp in cut_parameters[t]:
                print('{:>10} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8} {:>8}'.format('name', 'ubheu', 'dheu', 'theu', 'ubbb', 'lbbb', 'lbroot', 'lbroot*', 'dbb', 'nodes', 'tbb'))
                i, f = 0, 0
                while f < 10:
                    random.seed(seed[n]+i)
                    instance = MPVRPIR(**GetProblem(t,n))
                    s = SolveProblem(problem = instance, constructive = True, Model = False, name = '{}'.format(i), cutParameters = cp)
                    f += 1 if s != None else 0
                    i += 1

def OneInstance(type, n, i):
        print('Instance', i)
        random.seed(i)

        instanceArgs = GetProblem(type, n)
        print(instanceArgs)

        #instance = MPVRPIR(**instanceArgs) 
        #instance.SaveToFile('{}_{}_{}'.format(type, n, i))
        #instance = MPVRPIR()
        #instance.ReadInstance('{}_{}_{}'.format(type, n, i))

        SolveProblem(problem = instance, constructive = True, Model = False)

#import sys
#OneInstance(sys.argv[1], int(sys.argv[2]), int(sys.argv[3]))
#ForestProdExperiments()
#CutCalibration()

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

