from heapq import heappush, heappop, heapify

def ChooseTaskSmallestEndingTime(team, partialSolution, availableTasks, start, end):
    return min( availableTasks, key = lambda va: end(*va) )

def Constructive(P, ChooseTask = ChooseTaskSmallestEndingTime):
    # facilitating access to some structures
    teams = range(P.numTeams)
    customers = range(1, P.numVertices)
    tasks = [None] + [range(P.startingTask[v], P.numTasks) for v in customers]
    taskTime = P.taskTimes
    travelTime = P.travelTime

    teamType = []
    for k in range(P.numTeams):
        teamType.extend([k]*P.maxTeams[k])
    
    teamType.sort(key= lambda x: -P.teamProd[x])
    teamType = teamType[:P.maxTeamsAll]
    
    numTeams = 1
    teams = range(numTeams)

    while True: 
        # starting tasks
        available = {(v, P.startingTask[v]) for v in customers}

        solution = []

        # leftover task for each team
        leftover = [(None, None) for k in teams]
        
        # fraction of a task that is left to execute
        frac = {(v, a):1 for v in customers for a in tasks[v]}
        
        # unfinished tasks
        unfinished = set()

        currentDay = 0
        while available or unfinished:
            # tasks executed by each team on currentDay
            # list of (vertex, task, starting time, fraction) for each team
            solution.append( [[(0, None, 0, 0)] for k in teams] )

            # heap with events, an event is a (time of completion of some task, team, vertex, task)
            events = [(0, k, 0, None) for k in teams]
            
            # visited customers
            visited = set()

            while events:
                # next team completing a task, team is now free
                (t, k, v, a) = heappop(events)
               
                if v != 0 and a+1 < P.numTasks:
                    available.add( (v, a+1) )
                    
                start = lambda u, b: t + travelTime[ solution[currentDay][k][-1][0] ][u]
                end = lambda u, b: start(u,b) + taskTime[teamType[k]][u][b]*frac[(u,b)]

                # choose a task for the team
                if leftover[k] != (None, None):
                    v, a = leftover[k]
                else:
                    # determine possible tasks
                    availableForTeam = [(u, b) for (u, b) in available if t + travelTime[v][u] + travelTime[u][0] <= P.availableTime and u not in visited]
                    if v != 0 and a+1 < P.numTasks:
                        availableForTeam.append((v, a+1))

                    if not availableForTeam:
                        solution[-1][k].append((0, None, t + travelTime[ solution[currentDay][k][-1][0] ][0], 0))
                        continue

                    v, a = ChooseTask(teamType[k], solution, availableForTeam, start, end)
                
                    available.remove((v,a))
                visited.add(v)
                    
                # if it is not possible to finish the task 
                if end(v, a)+travelTime[v][0] > P.availableTime:
                    u = solution[currentDay][k][-1][0]
                    T = P.availableTime-travelTime[u][v]-travelTime[v][0]-t
                    f = T/taskTime[teamType[k]][v][a]

                    frac[(v, a)] -= f
                    leftover[k] = (v, a)
                    unfinished.add((v,a))
                    solution[-1][k].append((v,a, start(v,a), f))
                    solution[-1][k].append((0, None, P.availableTime, 0))

                else:
                    leftover[k] = None, None
                    if (v,a) in unfinished:
                        unfinished.remove((v,a))

                    heappush(events, (end(v, a), k, v, a))
                    solution[-1][k].append((v,a, start(v,a), frac[(v, a)]))

            currentDay += 1

        if len(solution) <= P.maxDays:
            '''
            print('feasible solution:', len(solution), 'days----------------')
            cost = 0
            for h in range(len(solution)):
                print('day', h, "----------------")
                for k, route in  enumerate(solution[h]):
                    if len(route) > 2:
                        print('route {} team {}: '.format(k, teamType[k]), end=' ')
                        print(route)
                        cost += P.teamCost[teamType[k]]
            print('total cost:', cost)
            '''

            cost = 0
            for h in range(len(solution)):
                for k, route in  enumerate(solution[h]):
                    if len(route) > 2:
                        cost += P.teamCost[teamType[k]]

            return solution, teamType[:numTeams], cost
            
        elif numTeams < P.maxTeamsAll:
            numTeams += 1
            teams = range(numTeams)
        else:
            #print('FAILED TO GENERATE FEASIBLE SOLUTION')
            return None

