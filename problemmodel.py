import numpy as np
import random
import math

class Problem:
    # Matrix nb(resources)xnb(features)
    # resources(i,j) = resource i has feature j
    resources = 0
    # Matrix nb(tasks)xnb(features)
    # needs(i,j) = task i requires feature j
    needs = 0
    # Matrix nb(tasks) x nb(schedule)
    # schedules(i,j) = task i is active at schedule j
    schedules = 0

    # Solution = matrix nb(task) x nb(resources)

    def __init__(self,resources = np.array([[]]),needs=np.array([[]]),schedules=np.array([[]]),values=np.array([])):
        self.resources = resources
        self.needs = needs
        self.schedules = schedules
        self.values = values
        self.compatibles = np.zeros((self.nb_tasks(),self.nb_resources()))

        for i in range(self.nb_tasks()):
            for r in range(self.nb_resources()):
                compatible = 1
                for f in range(self.nb_features()):
                    if self.needs[i,f] == 1 and self.resources[r,f] == 0:
                        compatible = 0
                        break
                self.compatibles[i,r] = compatible

    def nb_schedules(self):
        return self.schedules.shape[1]

    def nb_resources(self):
        return self.resources.shape[0]
    
    def nb_tasks(self):
        return self.needs.shape[0]

    def nb_features(self):
        return self.resources.shape[1]

    def task_features(self,i):
        """
        Return a list of the feature numbers needed by task i
        """
        return [index for index, value in enumerate(self.needs[i]) if value == 1]

    def resource_features(self,r):
        """
        Return a list of the feature numbers provided by resource r
        """
        return [index for index, value in enumerate(self.resources[r]) if value == 1]


    def check_solution(self,solution):
        """
        Check if a solution is valid
        Solution is a nparray nb(tasks) x nb(resources)
        """
        valid = True
        # Check if each resources are affected only once
        for s in range(self.nb_schedules()):
            for r in range(self.nb_resources()):
                # Get the sum of tasks with this resource on this schedule
                sum = 0
                for i in range(self.nb_tasks()):
                    sum += solution[i,r]*self.schedules[i,s]
                if sum > 1:
                    print("Resource "+str(r)+" is affected on "+str(sum)+" tasks for schedule "+str(s))
                    valid = False
        return valid

    def coverage(self,solution):
        """
        Compute the ratio of task that are assign to needed resources
        If a task is assigned to one or more resources but does not fulfill all
        its need, it is counted as not assigned
        """
        covered_tasks = 0
        covered_needs = solution @ self.resources
        for i in range(self.nb_tasks()):
            covered = True
            for f in range(self.nb_features()):
                if self.needs[i,f] > covered_needs[i,f]:
                    covered = False
                    break
            if covered:
                covered_tasks += 1
        return covered_tasks/self.nb_tasks()
    
class Solution:
    # the model this solution solve
    model = 0
    # Matrix nb(tasks) x nb(resources)
    solution = 0
    # the algorithm used to compute this solution
    algorithm = ""
    # time in ms to compute this solution
    time = 0

    # validity of the solution
    valid = True
    # coverage of the solution
    coverage = 0
    # value
    value = 0

    # coverage of needs for each tasks
    tasks_coverage = []
    tasks_features_covered = []

    def __init__(self, model, solution, algorithm, time):
        self.model = model
        self.solution = solution
        self.algorithm = algorithm
        self.time = time
        self.valid = model.check_solution(solution)
        self.value = 0
        self.coverage = model.coverage(solution)
        self.tasks_coverage = []
        self.tasks_features_covered = []

        for i in range(model.nb_tasks()):
            features = model.task_features(i)
            nb_feature_covered = 0
            features_covered = []
            for f in features:
                print(f"Search for (i,f)=({i},{f})")
                for r in range(model.nb_resources()):
                    if solution[i,r] == 1 and model.resources[r,f] == 1:
                        print(f"  -> found for resource {r}")
                        nb_feature_covered += 1
                        features_covered.append(f)
                        break
            if len(features) ==0:
                self.tasks_coverage.append(0)
            else:
                self.tasks_coverage.append(nb_feature_covered/len(features))
            self.tasks_features_covered.append(features_covered)
            if nb_feature_covered == len(features):
                self.value += model.values[i]

    def task_resources(self,i):
        """
        Return a list of the feature numbers needed by task i
        """
        return [index for index, value in enumerate(self.solution[i]) if value == 1]

    def __str__(self):
        return f"{self.algorithm} ({self.time}ms), score={math.floor(self.coverage*100)}%, valid={self.valid}\n"+str(self.solution)

def big_sample_problem():
    # 5 resources
    # 9 features
    # 5 schedules
    # 20 tasks
    resources = np.array([
        [1, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 1, 1, 1, 0, 1, 1, 0, 0],
        [1, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 1, 1, 1, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1]
    ])

    needs = np.array([
        [1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [0 , 0 , 0 , 1 , 0 , 0 , 0 , 1 , 0 ],
        [0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 ],
        [1 , 0 , 1 , 0 , 0 , 0 , 1 , 1 , 0 ],
        [1 , 0 , 0 , 0 , 0 , 1 , 0 , 1 , 0 ],
        [1 , 0 , 0 , 0 , 1 , 0 , 1 , 1 , 1 ],
        [0 , 1 , 0 , 0 , 1 , 1 , 0 , 1 , 0 ],
        [1 , 0 , 0 , 1 , 1 , 0 , 0 , 0 , 0 ],
        [0 , 1 , 0 , 0 , 0 , 1 , 1 , 1 , 1 ],
        [0 , 1 , 0 , 0 , 1 , 0 , 0 , 0 , 0 ],
        [1 , 1 , 0 , 0 , 1 , 0 , 0 , 1 , 0 ],
        [1 , 0 , 1 , 1 , 0 , 1 , 0 , 0 , 0 ],
        [0 , 1 , 1 , 0 , 0 , 0 , 0 , 1 , 0 ],
        [0 , 0 , 1 , 0 , 0 , 0 , 0 , 1 , 0 ],
        [0 , 0 , 1 , 1 , 0 , 1 , 0 , 0 , 1 ],
        [0 , 1 , 0 , 1 , 1 , 0 , 0 , 1 , 0 ],
        [1 , 0 , 0 , 1 , 0 , 1 , 0 , 1 , 1 ],
        [1 , 1 , 0 , 0 , 0 , 1 , 1 , 0 , 0 ],
        [0 , 1 , 0 , 1 , 1 , 0 , 1 , 1 , 1 ],
        [1 , 0 , 0 , 0 , 0 , 0 , 1 , 0 , 0 ]
    ])

    schedules = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 1, 0, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 1, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 1, 1],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 1],
        [0, 0, 0, 0, 0, 0, 1],
        [1, 0, 0, 0, 0, 0, 0],
        [0, 0, 0, 0, 1, 1, 0]
    ])
    values = [1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1,1]
    return Problem(resources=resources, needs=needs, schedules=schedules,values=values)


def random_problem(nb_tasks,nb_resources,nb_features,nb_schedules):
    resources = np.zeros((nb_resources, nb_features))
    for r in range(nb_resources):
        for f in range(nb_features):
            resources[r,f] = random.choices([1, 0], weights=[80, 20], k=1)[0]
    needs = np.zeros((nb_tasks, nb_features))
    for i in range(nb_tasks):
        for f in range(nb_features):
            needs[i,f] = random.choices([1, 0], weights=[20, 80], k=1)[0]
        needs[i,random.randint(0,nb_features-1)] = 1
    schedules = np.zeros((nb_tasks, nb_schedules))
    for i in range(nb_tasks):
        for s in range(nb_schedules):
            schedules[i,s] = random.choices([1, 0], weights=[50, 50], k=1)[0]
    values = np.zeros((nb_tasks))
    for i in range(nb_tasks):
        values[i] = random.randint(100,500)/100
    return Problem(resources=resources, needs=needs, schedules=schedules, values=values)
    
def small_sample_problem():
    # 5 resources
    # 9 features
    # 5 schedules
    # 20 tasks
    resources = np.array([
        [1, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 0, 0, 1, 1, 1]
    ])

    needs = np.array([
        [1 , 1 , 0 , 0 , 0 , 0 , 0 , 0 , 0 ],
        [0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 ],
        [0 , 0 , 0 , 0 , 0 , 0 , 0 , 1 , 1 ],
        [1 , 0 , 0 , 0 , 0 , 0 , 1 , 1 , 0 ],
        [0 , 0 , 0 , 0 , 1 , 1 , 0 , 0 , 0 ]
    ])

    schedules = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [0, 0, 1, 0, 0, 0, 0],
        [0, 0, 1, 1, 0, 0, 0]
    ])
    values = np.array([1,1,1,1,1])

    return Problem(resources=resources, needs=needs, schedules=schedules, values=values)


def super_easy_sample_problem():
    # 5 resources
    # 9 features
    # 5 schedules
    # 20 tasks
    resources = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 1]
    ])

    needs = np.array([
        [1 , 0, 0],
        [0 , 1, 0],
        [1 , 0, 1]
    ])

    schedules = np.array([
        [1, 0],
        [0, 1],
        [1, 1]
    ])
    values = np.array([1,1,1])
    return Problem(resources=resources, needs=needs, schedules=schedules, values=values)


def tiny_sample_problem():
    # 5 resources
    # 9 features
    # 5 schedules
    # 20 tasks
    resources = np.array([
        [1, 0, 0],
        [0, 1, 0],
        [0, 0, 1],
        [1, 1, 0],
        [0, 1, 1]
    ])

    needs = np.array([
        [1 , 1, 0 ],
        [0 , 1, 0 ],
        [0 , 0 , 1],
        [0 , 1 , 1]
    ])

    schedules = np.array([
        [1, 1, 0, 0, 0, 0, 0],
        [0, 1, 1, 0, 0, 0, 0],
        [1, 1, 1, 0, 0, 0, 0],
        [0, 1, 1, 1, 0, 0, 0]
    ])
    values = np.array([1,1,1,1])


    return Problem(resources=resources, needs=needs, schedules=schedules, values=values)
