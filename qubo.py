import problemmodel
import numpy as np

from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import RandomSampler
from dimod import ExactSolver
import dimod
import time

def index_to_qubomodel(model,i,r):
    return model.nb_resources()*i+r

def qubomodel_to_index(model,q):
    i = q//model.nb_resources()
    r = q%model.nb_resources()
    return (i,r)

def empty_qubo(model):
    return np.zeros((model.nb_tasks()*model.nb_resources(),model.nb_tasks()*model.nb_resources()))

def schedule_qubo(model):
    """
    Add the penality if the same resource is affected to two tasks at the same schedule
    """
    Q = empty_qubo(model)
    penalty = 5
    for s in range(model.nb_schedules()):
        for r in range(model.nb_resources()):
            for i in range(model.nb_tasks()):
                for j in range(i+1,model.nb_tasks()):
                    Q[index_to_qubomodel(model,i,r),index_to_qubomodel(model,j,r)] += penalty*model.schedules[i,s]*model.schedules[j,s]
                    Q[index_to_qubomodel(model,j,r),index_to_qubomodel(model,i,r)] += penalty*model.schedules[i,s]*model.schedules[j,s]
    return Q

def needs_qubo(model):
    """
    Add a bonus if a task fulfill all its needs.
    Problem with multiple resource assignment!
    """
    Q = empty_qubo(model)
    for f in range(model.nb_features()):
        for i in range(model.nb_tasks()):
            for r in range(model.nb_resources()):
                Q[index_to_qubomodel(model,i,r),index_to_qubomodel(model,i,r)] -= model.needs[i,f]*model.resources[r,f]
    return Q

def single_assignement_qubo(model):
    """
    Add a penality if multiple resources are assigned to the same task
    """
    Q = empty_qubo(model)
    penalty = 5
    for i in range(model.nb_tasks()):
        for r1 in range(model.nb_resources()):
            for r2 in range(r1+1,model.nb_resources()):
                Q[index_to_qubomodel(model,i,r1),index_to_qubomodel(model,i,r2)] += penalty
                Q[index_to_qubomodel(model,i,r2),index_to_qubomodel(model,i,r1)] += penalty
    return Q

def generate_qubo(model):
    return schedule_qubo(model) + needs_qubo(model) + single_assignement_qubo(model)

def qubo_solution_to_affectation_matrix(model,sample):
    return np.array(list(sample.values())).reshape(model.nb_tasks(),model.nb_resources())

def solve_with_exactSolver(model):
    Q = generate_qubo(model)
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
    start = time.time()
    sampleset = ExactSolver().sample(bqm)
    duration = (time.time() - start)*1000
    qubo_solution = sampleset.first.sample
    solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
    return problemmodel.Solution(model,solution,"Exact Solver",duration)

def solve_with_simulatedAnnealing(model):
    Q = generate_qubo(model)
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
    start = time.time()
    sampleset = SimulatedAnnealingSampler().sample(bqm,num_reads=1000, beta_schedule_type='linear', initial_state_generator="random")
    duration = (time.time() - start)*1000
    qubo_solution = sampleset.first.sample
    solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
    return problemmodel.Solution(model,solution,"Exact Solver",duration)