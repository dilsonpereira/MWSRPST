import random
import math

class MPVRPIR:
    def __init__(self, 
            numVertices = None, 
            numTasks = None, # List with the number of tasks in each service
            maxDays = None, # Maximum available number of days
            baseTimes = None, # Set for sampling task times
            numTeams = None, # Number of team types
            teamProd = None, # Productivity of each team type
            teamCost = None, # Cost of each team type
            maxTeamsAll = None, # Maximum number of teams overall
            maxTeams = None, # Maximum available number of teams of each type
            availableTime = 8, # Daily available time
            side = 10, # Size of the squere in which customers will be scattered
            speed = 40 # Vehicle speed in Km/h
            ):

        if numVertices != None:
            self.numVertices = numVertices
            self.numTasks = numTasks
            self.availableTime = availableTime
            self.maxDays = maxDays
            self.baseTimes = baseTimes
            self.side = side
            self.speed = speed
            self.teamProd = teamProd
            self.teamCost = teamCost
            self.maxTeamsAll = maxTeamsAll
            self.maxTeams = maxTeams
            self.numTeams = numTeams

            self.GenerateCustomerGraph()
            [random.choice([1]) for u in range(self.numVertices)]

            self.GenerateTaskTimes()

            self.startingTask = [None]+[random.choice(range(self.numTasks)) for u in range(1, self.numVertices)]
            self.startingTask = [None]+[0 for u in range(1, self.numVertices)]

    def GenerateCustomerGraph(self):
        side = self.side
        speed = self.speed
        n = self.numVertices

        p = [(random.random()*side, random.random()*side) for i in range(n)]

        d = lambda p1, p2: math.sqrt((p1[0]-p2[0])**2 + (p1[1]-p2[1])**2)

        self.travelTime = [[0 for u in range(n)] for v in range(n)]
        for u in range(n):
            for v in range(u+1, n):
                self.travelTime[u][v] = self.travelTime[v][u] =  d(p[u], p[v])/speed

    def GenerateTaskTimes(self):
        # std task times per ha
        self.stdTimes = [random.choice(self.baseTimes) for a in range(self.numTasks)]

        # each customer has a size, a customer with size 1, will have times as in stdTimes
        # a customer with size 2 will have times 2*times, and so on
        self.size = [None] + [0.5+random.random()*1.5 for c in range(1, self.numVertices)]

        self.GenerateTaskTimesb()

    def GenerateTaskTimesb(self):
        self.taskTimes = [ [None] + [ [ 0 for a in range(self.numTasks) ] for c in range(1,self.numVertices) ] for u in range(self.numTeams) ]
        for u in range(self.numTeams):
            for c in range(1, self.numVertices):
                for a in range(self.numTasks):
                    self.taskTimes[u][c][a] = self.stdTimes[a]*self.size[c]/self.teamProd[u]

    def SaveToFile(self, fileName):
        f = open(fileName, 'w')

        f.write('Number_of_vertices: {}\n'.format(self.numVertices))
        f.write('Number_of_tasks: {}\n'.format(self.numTasks))
        f.write('Number_of_team_types: {}\n'.format(self.numTeams))
        f.write('Daily_available_time: {}\n'.format(self.availableTime))
        f.write('Number_of_days: {}\n'.format(self.maxDays))
        f.write('Travel_times:\n')
        for v in range(self.numVertices):
            f.write(' '.join(['{0:.3f}'.format(x) for x in self.travelTime[v]]) + '\n')
        f.write('Standard_task_times:\n')
        f.write(' '.join(['{0:.3f}'.format(x) for x in self.stdTimes]) + '\n')
        f.write('Customer_sizes:\n')
        f.write('-1 ' + ' '.join(['{0:.3f}'.format(x) for x in self.size[1:]]) + '\n')
        f.write('Starting_tasks:\n')
        f.write('-1 ' + ' '.join(['{}'.format(0 if x == None else x) for x in self.startingTask[1:]]) + '\n')
        f.write('Team_proficiencies:\n')
        f.write(' '.join(['{0:.3f}'.format(x) for x in self.teamProd]) + '\n')
        f.write('Team_costs:\n')
        f.write(' '.join(['{0:.3f}'.format(x) for x in self.teamCost]) + '\n')
        f.write('Team_availabilities:\n')
        f.write(' '.join(['{}'.format(x) for x in self.maxTeams]) + '\n')
        f.write('Max_teams: {}\n'.format(self.maxTeamsAll))

        f.close()

    def ReadInstance(self, fileName):
        f = open(fileName, 'r')

        s = f.readline().split()
        self.numVertices = int(s[1])

        s = f.readline().split()
        self.numTasks = int(s[1])

        s = f.readline().split()
        self.numTeams = int(s[1])

        s = f.readline().split()
        self.availableTime = int(s[1])

        s = f.readline().split()
        self.maxDays = int(s[1])

        self.travelTime = []
        f.readline()
        for a in range(self.numVertices):
            self.travelTime.append([float(x) for x in f.readline().split()])

        f.readline()
        self.stdTimes = [float(x) for x in f.readline().split()]

        f.readline()
        self.size = [float(x) for x in f.readline().split()]
        self.size[0] = None

        f.readline()
        self.startingTask = [int(x) for x in f.readline().split()]
        self.startingTask[0] = None

        f.readline()
        self.teamProd = [float(x) for x in f.readline().split()]

        f.readline()
        self.teamCost = [float(x) for x in f.readline().split()]

        f.readline()
        self.maxTeams = [int(x) for x in f.readline().split()]

        s = f.readline().split()
        self.maxTeamsAll = int(s[1])

        self.GenerateTaskTimesb()

        f.close()

