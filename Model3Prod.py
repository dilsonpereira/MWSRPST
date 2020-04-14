from MPVRPIR import *
from math import ceil
from collections import namedtuple, deque

import random

import cplex
from cplex.exceptions import CplexError
from cplex.callbacks import HeuristicCallback, MIPInfoCallback, UserCutCallback
from math import ceil, floor

import Constructive

from collections import defaultdict

import numpy as np

import os
import psutil

class MyCutCallback(UserCutCallback):
    def __init__(self, env):
        UserCutCallback.__init__(self, env)

    def GetValues(self):
        vars = self.X + self.Y + self.F
        vals = self.get_values(vars)
        val = dict(zip(vars, vals))
        self.val = val
        P = self.P

        self.x_ = np.array([[[[val[self.x(k, h, u, v)] for v in range(P.numVertices)] for u in range(P.numVertices)] for h in range(P.maxDays)] for k in range(P.numTeams)])
        self.y_ = np.array([[[[val[self.y(k, h, i, v)] for v in range(P.numVertices)] for i in range(P.numTasks)] for h in range(P.maxDays)] for k in range(P.numTeams)])
        self.f_ = np.array([[[[val[self.f(k, h, i, v)] for v in range(P.numVertices)] for i in range(P.numTasks)] for h in range(P.maxDays)] for k in range(P.numTeams)])

    def AddCut(self, coefs, sense, rhs):
        cut = list(zip(*coefs))
        #self.add(cut=cut, sense=sense, rhs=rhs, use=self.use_cut.filter)
        self.add(cut=cut, sense=sense, rhs=rhs)


    def __call__(self):
        self.GetValues()

        for s in self.SP:
            if s['active']:
                C = s['separation']()
                for coefs, sense, rhs, vio, in C:
                    if vio > s['threshold']:
                        s['numcuts'] += 1
                        self.AddCut(coefs, sense, rhs)
        
    def CutProposition2(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                for v in range(1, P.numVertices):
                    for a in range(P.startingTask[v], P.numTasks):
                        for b in range(a+1, P.numTasks):
                            for c in range(b+1, P.numTasks):
                                vio = y_[k, h, a, v] + y_[k, h, c, v] - f_[k, h, b, v] - 1.0
                                if vio > 1e-6:
                                    coefs = [(y(k, h, a, v), 1), 
                                             (y(k, h, c, v), 1),
                                             (f(k, h, b, v), -1)]

                                    cuts.append((coefs, 'L', 1, vio))
        return cuts

    def CutProposition3(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                for a in range(P.startingTask[v], P.numTasks):
                    for b in range(a+1, P.numTasks):
                        for c in range(b+1, P.numTasks):
                            vio = -1 
                            coefs = []
                            for k in range(P.numTeams):
                                vio += y_[k, h, a, v] + y_[k, h, c, v] - f_[k, h, b, v]
                                coefs += [(y(k, h, a, v), 1), 
                                         (y(k, h, c, v), 1),
                                         (f(k, h, b, v), -1)]

                            if vio > 1e-6:
                                cuts.append((coefs, 'L', 1, vio))

        return cuts


    def CutProposition4(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_

        for h in range(P.maxDays):
            for h2 in range(h+1, P.maxDays):
                for v in range(1, P.numVertices):
                    for a in range(P.startingTask[v], P.numTasks):
                        for b in range(a+1, P.numTasks):
                            vio = -1 
                            coefs = []
                            for k in range(P.numTeams):
                                vio += y_[k, h, b, v] 
                                coefs.append( (y(k, h, b, v), 1) )
                                vio += y_[k, h2, a, v] 
                                coefs.append( (y(k, h2, a, v), 1) )

                            if vio > 1e-6:
                                cuts.append((coefs, 'L', 1, vio))

        return cuts

    def CutProposition5(self):
        cuts = []
        P = self.P
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                for a in range(P.startingTask[v], P.numTasks):
                    for b in range(a+1, P.numTasks):
                        vio = -1
                        coefs = []
                        for h1 in range(h+1):
                            for k in range(P.numTeams):
                                vio += f_[k, h1, b, v] 
                                coefs.append( (f(k, h1, b, v), 1) )

                        for h2 in range(h+1, P.maxDays):
                            for k in range(P.numTeams):
                                vio += f_[k, h2, a, v] 
                                coefs.append( (f(k, h2, a, v), 1) )

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 1, vio))

        return cuts

    def CutProposition6(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for k in range(P.numTeams):
            for h in range(P.maxDays):
                for v in range(1, P.numVertices):
                    dT = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                    for a in range(P.startingTask[v], P.numTasks):
                        T = P.taskTimes[k][v][a]

                        if T <= dT:
                            continue

                        vio = y_[k, h, a, v]
                        coefs = [(y(k, h, a, v), 1)]

                        if h > 0:
                            vio -= y_[k, h-1, a, v]
                            coefs.append( ( y(k, h-1, a, v), -1) )

                        if h < P.maxDays-1:
                            vio -= y_[k, h+1, a, v]
                            coefs.append( ( y(k, h+1, a, v), -1) )

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        return cuts

    def CutProposition13(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        x, x_ = self.x, self.x_

        trios = [((u, a), (v, b), (w, c))
                for u in range(1, P.numVertices) for a in range(P.startingTask[u], P.numTasks)
                for v in range(u, P.numVertices) for b in range(P.startingTask[v], P.numTasks) if v != u or b > a
                for w in range(v, P.numVertices) for c in range(P.startingTask[w], P.numTasks) if (w != u and w != v) or (w == u and w == v  and c > b)]

        trios_ = []
        for ((u, a), (v, b), (w, c)) in trios:
            d1u, du1 = P.travelTime[0][u], P.travelTime[u][0]
            d1v, dv1 = P.travelTime[0][v], P.travelTime[v][0]
            d1w, dw1 = P.travelTime[0][w], P.travelTime[w][0]
            duv, dvu = P.travelTime[u][v], P.travelTime[v][u]
            duw, dwu = P.travelTime[u][w], P.travelTime[w][u]
            dvw, dwv = P.travelTime[v][w], P.travelTime[w][v]
            c1 = d1u + duv + dvw + dw1 
            c2 = d1u + duw + dwv + dv1
            c3 = d1v + dvu + duw + dw1
            c4 = d1v + dvw + dwu + du1
            c5 = d1w + dwu + duv + dv1
            c6 = d1w + dwv + dvu + du1
            d = min(c1, c2, c3, c4, c5, c6)
            trios_.append(((u, a), (v, b), (w, c), P.availableTime-d))

        trios = trios_

        for k in range(P.numTeams):
            for h in range(P.maxDays):
                xvars = [(k, h, 0, v) for v in range(1, P.numVertices)]
                coefsx = [(x(k, h, u, v), -1) for (k, h, u, v) in xvars]
                sumx = sum([ x_[k, h, u, v] for (k, h, u, v) in xvars ])

                for (u, a), (v, b), (w, c), d in trios:
                    if not (P.taskTimes[k][u][a] > d and P.taskTimes[k][v][b] > d  and P.taskTimes[k][w][c] > d):
                        continue

                    vio = y_[k, h, a, u] + y_[k, h, b, v] + y_[k, h, c, w] - sumx -1
                    if vio > 1e-6:
                        coefs = [( y(k, h, a, u), 1 ), (y(k, h, b, v), 1), (y(k, h, c, w), 1)]  + coefsx
                        cuts.append((coefs, 'L', 1, vio))

        return cuts

    def CutProposition8(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        x, x_ = self.x, self.x_

        for k in range(P.numTeams):
            for h in range(P.maxDays):
                for v in range(1, P.numVertices):
                    dT = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]

                    sumx = sum(x_[k, h, u, v] for u in range(P.numVertices) if u != v)
                    coefsx = [(x(k, h, u, v), -1) for u in range(P.numVertices) if u != v]
                    for a in range(P.startingTask[v], P.numTasks):
                        for b in range(a, P.numTasks):
                            vio = -sumx
                            coefs = []

                            T = sum(P.taskTimes[k][v][c] for c in range(a, b+1))
                            if T >= dT:
                                if h == 0 and b < P.numTasks-1:
                                    vio += y_[k, h, a, v] + y_[k, h, b+1, v]
                                    coefs.append((y(k, h, a, v), 1))
                                    coefs.append((y(k, h, b+1, v), 1))

                                elif h == P.maxDays-1 and a > P.startingTask[v]:
                                    vio += y_[k, h, a-1, v] + y_[k, h, b, v]
                                    coefs.append((y(k, h, a-1, v), 1))
                                    coefs.append((y(k, h, b, v), 1))

                                elif a > P.startingTask[v] and b < P.numTasks-1:
                                    vio += y_[k, h, a-1, v] + y_[k, h, b+1, v]
                                    coefs.append((y(k, h, a-1, v), 1))
                                    coefs.append((y(k, h, b+1, v), 1))

                                if vio > 1e-6:
                                    cuts.append((coefs+coefsx, 'L', 0, vio))

        return cuts

    def CutProposition9a(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for k in range(P.numTeams):
                for v in range(1, P.numVertices):
                    for a in range(P.startingTask[v], P.numTasks):
                        vio = y_[k, h, a, v] -1
                        coefs = [(y(k, h, a, v), 1)]
                        for k2 in range(P.numTeams):
                            if k == k2:
                                continue
                            for h2 in range(P.maxDays):
                                vio += f_[k2, h2, a, v] 
                                coefs.append((f(k2, h2, a, v), 1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 1, vio))

        return cuts

    def CutProposition9b(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for k in range(P.numTeams):
                for v in range(1, P.numVertices):
                    for a in range(P.startingTask[v], P.numTasks):
                        vio = -1
                        coefs = []
                        for k2 in range(P.numTeams):
                            if k == k2:
                                continue
                            vio += y_[k2, h, a, v] 
                            coefs.append((y(k2, h, a, v), 1))

                        for h2 in range(P.maxDays):
                            vio += f_[k, h2, a, v] 
                            coefs.append((f(k, h2, a, v), 1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 1, vio))

        return cuts

    def CutProposition7(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        x, x_ = self.x, self.x_

        for v in range(1, P.numVertices):
            dT = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
            for k in range(P.numTeams):
                for a in range(P.startingTask[v], P.numTasks):
                    T = P.taskTimes[k][v][a]
                    for b in range(a+1, P.numTasks):
                        T += P.taskTimes[k][v][b]
                        if T > dT:
                            for h in range(P.maxDays):
                                vio = y_[k, h, a, v] + y_[k, h, b, v]
                                coefs = [(y(k, h, a, v), 1),
                                        (y(k, h, b, v), 1)]

                                if h > 0:
                                    vio -= y_[k, h-1, a, v]
                                    coefs.append((y(k, h-1, a, v), -1))
                                if h < P.maxDays-1:
                                    vio -= y_[k, h+1, b, v]
                                    coefs.append((y(k, h+1, b, v), -1))

                                for u in range(P.numVertices):
                                    if u == v: 
                                        continue
                                    vio -= x_[k, h, u, v]
                                    coefs.append((x(k, h, u, v), -1))
                    
                                if vio > 1e-6:
                                    cuts.append((coefs, 'L', 0, vio))

        return cuts


    def CutProposition10(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_

        for v in range(1, P.numVertices):
            T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
            for a in range(P.startingTask[v], P.numTasks):
                vio = 1
                coefs = []

                for k in range(P.numTeams):
                    min = ceil(P.taskTimes[k][v][a]/T)
                    for h in range(P.maxDays):
                        vio -= y_[k, h, a, v]/min
                        coefs.append((y(k, h, a, v), 1/min))

                if vio > 1e-6:
                    cuts.append((coefs, 'G', 1, vio))

        return cuts

    def CutProposition11a(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        vio = (minh-1)*y_[k, h, a, v] 
                        coefs = [(y(k, h, a, v), (minh-1))]

                        for h_ in range(max(0, h-minh+1), h):
                            vio -= y_[k, h_, a, v]
                            coefs.append((y(k, h_, a, v), -1))

                        for h_ in range(h+1, min(P.maxDays, h+minh)):
                            vio -= y_[k, h_, a, v]
                            coefs.append((y(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        return cuts

    def CutProposition11b(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        if h > 0:
                            vio = (minh-1)*(y_[k, h, a, v]-y_[k, h-1, a, v])
                            coefs = [(y(k, h, a, v), (minh-1)),  (y(k, h-1, a, v), -(minh-1))]
                        else:
                            vio = (minh-1)*y_[k, h, a, v]
                            coefs = [(y(k, h, a, v), (minh-1))]

                        for h_ in range(h+1, min(P.maxDays, h+minh)):
                            vio -= y_[k, h_, a, v]
                            coefs.append((y(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        if h < P.maxDays-1:
                            vio = (minh-1)*(y_[k, h, a, v]-y_[k, h+1, a, v])
                            coefs = [(y(k, h, a, v), (minh-1)), (y(k, h+1, a, v), -(minh-1))]
                        else:
                            vio = (minh-1)*y_[k, h, a, v] 
                            coefs = [(y(k, h, a, v), (minh-1))]

                        for h_ in range(max(0, h-minh+1), h):
                            vio -= y_[k, h_, a, v]
                            coefs.append((y(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        return cuts

    def CutProposition12a(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        vio = y_[k, h, a, v] 
                        coefs = [(y(k, h, a, v), 1)]

                        for h_ in range(max(0, h-minh), min(P.maxDays, h+minh+1)):
                            vio -= f_[k, h_, a, v]
                            coefs.append((f(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        return cuts


    def CutProposition12b(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        f, f_ = self.f, self.f_

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        if h > 0:
                            vio = y_[k, h, a, v]-y_[k, h-1, a, v]
                            coefs = [(y(k, h, a, v), 1),  (y(k, h-1, a, v), -1)]
                        else:
                            vio = y_[k, h, a, v]
                            coefs = [(y(k, h, a, v), 1)]

                        for h_ in range(h, min(P.maxDays, h+minh+1)):
                            vio -= f_[k, h_, a, v]
                            coefs.append((f(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        for h in range(P.maxDays):
            for v in range(1, P.numVertices):
                T = P.availableTime-P.travelTime[0][v]-P.travelTime[v][0]
                for a in range(P.startingTask[v], P.numTasks):
                    for k in range(P.numTeams):
                        minh = ceil(P.taskTimes[k][v][a]/T)

                        if h < P.maxDays-1:
                            vio = y_[k, h, a, v]-y_[k, h+1, a, v]
                            coefs = [(y(k, h, a, v), 1), (y(k, h+1, a, v), -1)]
                        else:
                            vio = y_[k, h, a, v] 
                            coefs = [(y(k, h, a, v), 1)]

                        for h_ in range(max(0, h-minh), h+1):
                            vio -= f_[k, h_, a, v]
                            coefs.append((f(k, h_, a, v), -1))

                        if vio > 1e-6:
                            cuts.append((coefs, 'L', 0, vio))

        return cuts

    def CutProposition1(self):
        cuts = []
        P = self.P
        y, y_ = self.y, self.y_
        x, x_ = self.x, self.x_

        for h in range(P.maxDays):
            for k in range(P.numTeams):
                cap = [[x_[k, h, i, j] if i != j else 0 for j in range(P.numVertices)] for i in range(P.numVertices)]

                for v in range(1, P.numVertices):
                    if not any( (y_[k, h, a, v] > 1e-6 for a in range(P.startingTask[v], P.numTasks) ) ):
                        continue

                    mf, S = self.MaxFlow(0, v, cap)

                    W = set(range(P.numVertices))-S
                    coefs = []
                    for i in S:
                        for j in W:
                            coefs.append((x(k, h, i, j), 1))

                    for a in range(P.startingTask[v], P.numTasks):
                        vio = y_[k, h, a, v] - mf
                        if vio > 1e-6:
                            coefs_ = coefs[:]
                            coefs_.append((y(k, h, a, v), -1))

                            cuts.append((coefs_, 'G', 0, vio))

        return cuts

    def MaxFlow(self, s, t, cap):
        P = self.P

        n = P.numVertices
        flow = [[0 for j in range(n)] for i in range(n)]

        maxFlow = 0

        pathFound = True
        while pathFound:
            # S from the S-T cut induced by the saturated edges
            S = {s}
            Q = deque([s]) 

            # holds a tuple (father, flow, arc, direction) attached to each vertex v reachable in the residual graph, flow is the maximum flow that can be sent from s to v
            label = [(None, 1, None, None, None) for v in range(n)] 

            pathFound = False

            while Q and not pathFound:
                u = Q.popleft()

                for v in range(n):
                    if v == u or v in S:
                        continue

                    if cap[u][v] - flow[u][v] > flow[v][u]:
                        g, i, j, d = cap[u][v] - flow[u][v], u, v, 1
                    else:
                        g, i, j, d = flow[v][u], v, u, -1

                    if g:
                        Q.append(v)
                        S.add(v)
                        label[v] = (u, min(label[u][1], g), i, j, d)
                        if v == t:
                            pathFound = True
                            break

            if pathFound:
                v, (u, mf, i, j, d) = t, label[t]
                while u != None:
                    flow[i][j] += d*mf

                    v, (u, _, i, j, d) = u, label[u]

                maxFlow += mf

        return maxFlow, S

    def CreateCutList(self, cutParameters= None):
        self.SP = []

        if not cutParameters:
            s = {'separation':self.CutProposition2, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition3, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition4, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition5, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition6, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition13, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition8, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition9a, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition9b, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition7, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition10, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition11a, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition11b, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition12a, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition12b, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)

            s = {'separation':self.CutProposition1, 'numcuts':0, 'active':True, 'threshold':1e-6}
            self.SP.append(s)
        else:
            s = {'separation':self.CutProposition2, 'numcuts':0, 'active':cutParameters['CutProposition2']['active'], 'threshold':cutParameters['CutProposition2']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition3, 'numcuts':0, 'active':cutParameters['CutProposition3']['active'], 'threshold':cutParameters['CutProposition3']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition4, 'numcuts':0, 'active':cutParameters['CutProposition4']['active'], 'threshold':cutParameters['CutProposition4']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition5, 'numcuts':0, 'active':cutParameters['CutProposition5']['active'], 'threshold':cutParameters['CutProposition5']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition6, 'numcuts':0, 'active':cutParameters['CutProposition6']['active'], 'threshold':cutParameters['CutProposition6']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition13, 'numcuts':0, 'active':cutParameters['CutProposition13']['active'], 'threshold':cutParameters['CutProposition13']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition8, 'numcuts':0, 'active':cutParameters['CutProposition8']['active'], 'threshold':cutParameters['CutProposition8']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition9a, 'numcuts':0, 'active':cutParameters['CutProposition9a']['active'], 'threshold':cutParameters['CutProposition9a']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition9b, 'numcuts':0, 'active':cutParameters['CutProposition9b']['active'], 'threshold':cutParameters['CutProposition9b']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition7, 'numcuts':0, 'active':cutParameters['CutProposition7']['active'], 'threshold':cutParameters['CutProposition7']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition10, 'numcuts':0, 'active':cutParameters['CutProposition10']['active'], 'threshold':cutParameters['CutProposition10']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition11a, 'numcuts':0, 'active':cutParameters['CutProposition11a']['active'], 'threshold':cutParameters['CutProposition11a']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition11b, 'numcuts':0, 'active':cutParameters['CutProposition11b']['active'], 'threshold':cutParameters['CutProposition11b']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition12a, 'numcuts':0, 'active':cutParameters['CutProposition12a']['active'], 'threshold':cutParameters['CutProposition12a']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition12b, 'numcuts':0, 'active':cutParameters['CutProposition12b']['active'], 'threshold':cutParameters['CutProposition12b']['e']}
            self.SP.append(s)

            s = {'separation':self.CutProposition1, 'numcuts':0, 'active':cutParameters['CutProposition1']['active'], 'threshold':cutParameters['CutProposition1']['e']}
            self.SP.append(s)

class MyInfo(MIPInfoCallback):
    def __init__(self, env):
        MIPInfoCallback.__init__(self, env)
        self.info = defaultdict(lambda: 0)
    def __call__(self):
        self.info['lb'] = self.get_best_objective_value()
        self.info['numnodes'] = self.get_num_nodes()
        if self.get_num_nodes() == 0:
            self.info['lbroot'] = self.get_best_objective_value()

        process = psutil.Process(os.getpid())

        if process.memory_percent() > 90:
            print('out of memory')
            self.abort()

def SolveCplexModel3(P, heu=None, useHeuristicCallback=False, integer = True, cutParameters = None):
    prob = cplex.Cplex()
        
    prob.objective.set_sense(prob.objective.sense.minimize)

    # creating the variables
    Variable = namedtuple('Variable', ['name', 'obj', 'lb', 'ub', 'type'])

    # x[k,h,u,v] in {0,1}: 1 if team k goes from u to v on day h
    # x is a list of tuples, one for each variable x, each tuple consists of (name, obj, lb, ub, type)
    xname = lambda k,h,u,v: 'x_{}_{}_{}_{}'.format(k,h,u,v)
    x = [ Variable(name=xname(k,h,u,v), obj=0 if u != 0 else P.teamCost[k], lb=0, ub=1, type='I') for k in range(P.numTeams) for h in range(P.maxDays) for u in range(P.numVertices) for v in range(P.numVertices) ] 
    name, obj, lb, ub, types = list(zip(*x))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # q[k,h,u,v] in R+: moment in which team k arrives at vertex v comming from u on day h
    qname = lambda k,h,u,v: 'q_{}_{}_{}_{}'.format(k,h,u,v)
    q = [ Variable(name=qname(k,h,u,v), obj=0, lb=0, ub=cplex.infinity, type='C') for k in range(P.numTeams) for h in range(P.maxDays) for u in range(P.numVertices) for v in range(P.numVertices) ]
    name, obj, lb, ub, types = list(zip(*q))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # y[k,h,i,v] in {0,1}: 1 if team k executes task i of customer v on day h
    yname = lambda k,h,i,v: 'y_{}_{}_{}_{}'.format(k,h,i,v)
    y = [ Variable(name=yname(k,h,i,v), obj=0, lb=0, ub=1, type='I') for k in range(P.numTeams) for h in range(P.maxDays) for i in range(P.numTasks) for v in range(P.numVertices) ]
    name, obj, lb, ub, types = list(zip(*y))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
        prob.order.set([(var.name, 10, prob.order.branch_direction.default) for var in y])
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # f[k,h,i,v] in R+: fraction of task i of customer v executed by team k on day h
    fname = lambda k,h,i,v: 'f_{}_{}_{}_{}'.format(k,h,i,v)
    f = [ Variable(name=fname(k,h,i,v), obj=0, lb=0, ub=1, type='C' ) for k in range(P.numTeams) for h in range(P.maxDays) for i in range(P.numTasks) for v in range(P.numVertices) ]
    name, obj, lb, ub, types = list(zip(*f))
    if integer:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub, types = types)
    else:
        prob.variables.add(names = name, obj = obj, lb = lb, ub = ub)

    # Every task must be executed in total
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v], P.numTasks):
            terms = []
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    terms.append((1, fname(k, h, i, v)))
            coefs, vars = list(zip(*terms))
            prob.linear_constraints.add(lin_expr = [[vars,coefs]], senses = ['E'], rhs = [1])


    # If a fraction of a task is executed on a period, than the corresponding task must be 'activated' on that period
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v], P.numTasks):
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    M = min((P.availableTime - P.travelTime[0][v] - P.travelTime[v][0])/P.taskTimes[k][v][i], 1)
                    vars = [fname(k,h,i,v), yname(k,h,i,v)]
                    coefs = [1, -M]
                    prob.linear_constraints.add(lin_expr = [[vars, coefs]], senses = ['L'], rhs = [0])
    
    # An activity can only be executed in a place if the activity on which it depends has been completed before
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v]+1, P.numTasks):
            for h in range(P.maxDays):
                terms = []
                for k in range(P.numTeams):
                    terms.append((1, yname(k, h, i, v)))
                    for h2 in range(h+1):
                        terms.append((-1, fname(k,h2,i-1,v)))
                coefs, vars = zip(*terms)
                prob.linear_constraints.add(lin_expr=[[vars, coefs]], senses = ['L'], rhs = [0] )

    # An anactivity left unfinished on the previous day has to be the first to be executed on the following day
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v], P.numTasks):
            for k in range(P.numTeams):
                for h in range(1, P.maxDays):
                    terms = [(1, xname(k,h,0,v)), (-1, yname(k,h-1, i, v))]
                    for h2 in range(h):
                        terms.append((1, fname(k, h2, i, v)))
            
                    coefs, vars = zip(*terms)         
                    prob.linear_constraints.add(lin_expr=[[vars, coefs]], senses = ['G'], rhs = [0])

    # Only the last activity can be left unfinished
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v], P.numTasks):
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    terms = [(1, xname(k,h,v,0)), (-1, yname(k,h, i, v))]
                    for h2 in range(h+1):
                        terms.append((1, fname(k, h2, i, v)))
            
                    coefs, vars = zip(*terms)         
                    prob.linear_constraints.add(lin_expr=[[vars, coefs]], senses = ['G'], rhs = [0])

    # If an activity is executed in a place on a period, then the corresponding team has to visit the place
    for v in range(1, P.numVertices):
        for i in range(P.startingTask[v], P.numTasks):
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    vars, coefs = [yname(k,h,i,v)], [-1]

                    for u in range(P.numVertices):
                        if u == v: 
                            continue
                        vars.append(xname(k,h,u,v))
                        coefs.append(1)
        
                    prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['G'], rhs = [0])

    # A team arriving in a customer on a period must also leave the customer
    for v in range(P.numVertices):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [], []

                for u in range(P.numVertices):
                    if u == v: 
                        continue

                    vars.extend([xname(k,h,u,v), xname(k,h,v,u)])
                    coefs.extend([1,-1])
                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['E'], rhs = [0])

    # A customer cannot be visited more than once in a period
    for v in range(1, P.numVertices):
        for h in range(P.maxDays):
            vars, coefs = [], []

            for k in range(P.numTeams):
                for u in range(P.numVertices):
                    if u == v:
                        continue
                    vars.append(xname(k, h, u, v))
                    coefs.append(1)

            prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['L'], rhs = [1])
                
    # flow out of the depot
    for v in range(1, P.numVertices):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [xname(k,h,0,v), qname(k, h, 0, v)], [-P.travelTime[0][v], 1]

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['E'], rhs = [0])

    # flow conservation
    for v in range(1, P.numVertices):
        for k in range(P.numTeams):
            for h in range(P.maxDays):
                vars, coefs = [], []
                for u in range(P.numVertices):
                    if u == v:
                        continue
                    vars.extend( [qname(k, h, u, v), xname(k, h, v, u), qname(k, h, v, u)] )
                    coefs.extend( [1, P.travelTime[v][u], -1] )

                for a in range(P.startingTask[v], P.numTasks):
                    vars.append(fname(k, h, a, v))
                    coefs.append(P.taskTimes[k][v][a])

                prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['E'], rhs = [0])

    # flow capacity
    for v in range(P.numVertices):
        for u in range(P.numVertices):
            if u == v:
                continue
            for k in range(P.numTeams):
                for h in range(P.maxDays):
                    vars, coefs = [xname(k,h,u,v), qname(k, h, u, v)], [-P.availableTime, 1]

                    prob.linear_constraints.add(lin_expr=[[vars,coefs]], senses = ['L'], rhs = [0])
    
    # maximum number of teams of each type
    for k in range(P.numTeams):
        for h in range(P.maxDays):
            vars, coefs = [], []
            for v in range(1, P.numVertices):
                vars.append(xname(k,h,0,v))
                coefs.append(1)
            prob.linear_constraints.add(lin_expr = [[vars, coefs]], senses = ['L'], rhs = [P.maxTeams[k]])
    
    # maximum number of teams
    for h in range(P.maxDays):
        vars, coefs = [], []
        for k in range(P.numTeams):
            for v in range(1, P.numVertices):
                vars.append(xname(k,h,0,v))
                coefs.append(1)
        prob.linear_constraints.add(lin_expr = [[vars, coefs]], senses = ['L'], rhs = [P.maxTeamsAll])

    prob.parameters.preprocessing.presolve.set(0)

    # create mipstart solution
    if integer and heu:
        mipstart = Model3MipStartFromHeuristic(P, heu, x+q+y+f, xname, qname, yname, fname)
        prob.MIP_starts.add(mipstart, prob.MIP_starts.effort_level.check_feasibility)

    # register heuristic callback
    if integer and useHeuristicCallback:
        heuCallback = prob.register_callback(Model3HeuristicCallback)
        heuCallback.P = P
        heuCallback.xname, heuCallback.yname, heuCallback.qname, heuCallback.fname = xname, yname, qname, fname
        heuCallback.vars = x+y+q+f

    useCuts = True
    if integer and useCuts:
        cc = prob.register_callback(MyCutCallback)
        cc.CreateCutList(cutParameters)
        cc.P = P
        cc.X = [ xname(k, h, u, v) for k in range(P.numTeams) for h in range(P.maxDays) for u in range(P.numVertices) for v in range(P.numVertices) ] 
        cc.Y = [ yname(k,h,i,v) for k in range(P.numTeams) for h in range(P.maxDays) for i in range(P.numTasks) for v in range(P.numVertices) ]
        cc.F = [ fname(k,h,i,v) for k in range(P.numTeams) for h in range(P.maxDays) for i in range(P.numTasks) for v in range(P.numVertices) ]
        cc.x, cc.y, cc.f = xname, yname, fname

    if integer:
        ic = prob.register_callback(MyInfo)
        info = ic.info
    else:
        info = {}

    #prob.parameters.mip.strategy.variableselect.set(3)
    #prob.parameters.mip.strategy.fpheur.set(-1)
    #prob.parameters.mip.strategy.heuristicfreq.set(-1)
    #prob.parameters.mip.strategy.lbheur.set(0)
    #prob.parameters.mip.strategy.rinsheur.set(-1)
    #prob.parameters.emphasis.mip.set(3)

    prob.parameters.mip.cuts.bqp.set(-1)
    prob.parameters.mip.cuts.cliques.set(-1)
    prob.parameters.mip.cuts.covers.set(-1)
    prob.parameters.mip.cuts.disjunctive.set(-1)
    prob.parameters.mip.cuts.flowcovers.set(-1)
    prob.parameters.mip.cuts.pathcut.set(-1)
    prob.parameters.mip.cuts.gomory.set(-1)
    prob.parameters.mip.cuts.gubcovers.set(-1)
    prob.parameters.mip.cuts.implied.set(-1)
    prob.parameters.mip.cuts.localimplied.set(-1)
    prob.parameters.mip.cuts.liftproj.set(-1)
    prob.parameters.mip.cuts.mircut.set(-1)
    prob.parameters.mip.cuts.mcfcut.set(-1)
    prob.parameters.mip.cuts.rlt.set(-1)
    prob.parameters.mip.cuts.zerohalfcut.set(-1)

    prob.set_log_stream(None)
    prob.set_results_stream(None)
    prob.set_error_stream(None)
    prob.set_warning_stream(None)

    prob.parameters.clocktype.set(2) # wall clock time
    prob.parameters.timelimit.set(3600)

    prob.solve()

    if prob.solution.status[prob.solution.get_status()] == 'MIP_infeasible':
        info['status'] = 'infeasible'
        return info, None

    if prob.solution.status[prob.solution.get_status()] == 'MIP_time_limit_infeasible':
        info['status'] = 'time_limit_infeasible'
        return info, None

    ub = prob.solution.get_objective_value()

    if integer:
        info['ub'] = ub
        info['ubheu'] = len(heu) if heu != None else None
        info['numnodes'] += 1
        info['status'] = prob.solution.status[prob.solution.get_status()]
    else:
        info['lb'] = ub

    if integer:
        solution = []
        used = set()
        for h in range(P.maxDays):
            solution.append( [[] for k in range(P.numTeams)] )
          
            for k in range(P.numTeams):
                solution[-1][k].append( [(0, None, 0, 0)] )

                s = 0
                while True: 
                    t = None
                    for v in range(P.numVertices):
                        if not (k, h, s, v) in used:
                            vx = prob.solution.get_values(xname(k,h,s,v))
                            if vx > 0:
                                t = v
                                break
                    if t == None:
                        break
                    used.add( (k, h, s, t) )
                    q = prob.solution.get_values(qname(k, h, s, t))
                    s = t
                    if s:
                        ta = 0
                        for i in range(P.startingTask[s], P.numTasks):
                            vf = prob.solution.get_values(fname(k,h,i,s))
                            ta += vf*P.taskTimes[k][s][i]
                            if vf > 0:
                                solution[-1][k][-1].append((s, i, vf, q+ta))
                                T = P.availableTime-P.travelTime[0][s]-P.travelTime[s][0]
                    else:
                        solution[-1][k][-1].append((0, None, 0, q))
                        solution[-1][k].append( [(0, None, 0, 0)] )

        for h in range(P.maxDays-1, -1, -1):
            if all([len(solution[h][k]) == 1 for k in range(P.numTeams)]):
                solution.pop(h)
    else:
        solution = None

    return info, solution

class Model3HeuristicCallback(HeuristicCallback):
    def __call__(self):
        cur = self.get_incumbent_objective_value()

        varnames = [var.name for var in self.vars]
        varvalues = self.get_values(varnames)
        
        vardict = dict(zip(varnames, varvalues))
        def ChooseTaskHeuristicCallback(team, partialSolution, availableTasks, start, end):
            currentDay = len(partialSolution)-1
            if currentDay >= self.P.maxDays:
                return availableTasks[0]

            fval = lambda v, a: vardict[ self.fname(team, currentDay, a, v) ]

            return max( availableTasks, key = lambda va: fval(*va) )

        heu = Constructive.Constructive(self.P, ChooseTaskHeuristicCallback)
        if heu != None and heu[2] < cur:
            solution = Model3MipStartFromHeuristic( self.P, heu, self.vars, self.xname, self.qname, self.yname, self.fname)
            self.set_solution(solution)

def Model3MipStartFromHeuristic(P, heu, vars, xname, qname, yname, fname):
    # heu is a list of days
    # each day is a list of teams
    # each team is a list of activities
    # each task is a tuple (vertex, task, starting time)

    heu, type, cost = heu
                  
    mipdict = {var.name:0 for var in vars}              
    for h in range(len(heu)):
        for k in range(len(heu[h])):
            t = 0
            for (v1, a1, s1, f1), (v2, a2, s2, f2) in zip(heu[h][k], heu[h][k][1:]):
                if v2 != v1:
                    mipdict[xname(type[k], h, v1, v2)] = 1 
                if v2 != v1:
                    mipdict[qname(type[k], h, v1, v2)] = s2
                if v2 != 0:
                    mipdict[yname(type[k], h, a2, v2)] = 1
                if v2 != 0:
                    mipdict[fname(type[k], h, a2, v2)] = f2
    return list(zip(*mipdict.items()))

 
def Solve(problem, initialSolution = None, integer = True, cutParameters = None):
    return SolveCplexModel3(problem, initialSolution, useHeuristicCallback = True, integer = integer, cutParameters = cutParameters)



