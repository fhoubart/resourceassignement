import numpy as np

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

    def __init__(self,resources = np.array([[]]),needs=np.array([[]]),schedules=np.array([[]])):
        self.resources = resources
        self.needs = needs
        self.schedules = schedules

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

    # coverage of needs for each tasks
    tasks_coverage = []
    tasks_features_covered = []

    def __init__(self, model, solution, algorithm, time):
        self.model = model
        self.solution = solution
        self.algorithm = algorithm
        self.time = time
        self.valid = model.check_solution(solution)
        self.coverage = model.coverage(solution)
        for i in range(model.nb_tasks()):
            features = model.task_features(i)
            nb_feature_covered = 0
            features_covered = []
            for f in features:
                for r in range(model.nb_resources()):
                    if solution[i,r] == 1 and model.resources[r,f] == 1:
                        nb_feature_covered += 1
                        features_covered.append(f)
                        break
            self.tasks_coverage.append(nb_feature_covered/len(features))
            self.tasks_features_covered.append(features_covered)

    def task_resources(self,i):
        """
        Return a list of the feature numbers needed by task i
        """
        return [index for index, value in enumerate(self.solution[i]) if value == 1]



def big_sample_problem():
    # 5 resources
    # 9 features
    # 5 schedules
    # 20 tasks
    resources = np.array([
        [0, 1, 1, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 1, 0, 1, 0, 0, 0],
        [1, 1, 0, 0, 1, 0, 1, 0, 0],
        [0, 1, 0, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 1, 1]
    ])

    needs = np.array([
        [1 , 1 , 0 , 1 , 0 , 0 , 0 , 0 , 0 ],
        [0 , 1 , 0 , 1 , 0 , 0 , 0 , 1 , 0 ],
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

    return Problem(resources=resources, needs=needs, schedules=schedules)


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

    return Problem(resources=resources, needs=needs, schedules=schedules)



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
    return Problem(resources=resources, needs=needs, schedules=schedules)


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


    return Problem(resources=resources, needs=needs, schedules=schedules)
