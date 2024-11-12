import problemmodel
import numpy as np

from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import RandomSampler
from dwave.system import DWaveSampler, EmbeddingComposite
from dimod import ExactSolver
import dimod
import time

"""
Constraints:
- for a given schedule, there should be no resources affected on two different tasks
- for a given task i and feature f, and for all resources r, z(i,f) >= x(i,r)*res(r,f)
    this constraints makes z(i,f) = max(x(i,f)*res(r,f)) for any r, making z(i,f)=1 if
    at least one resource with the given feature is affected at the task
"""
def index_to_qubomodel(model,i,r):
    return model.nb_resources()*i+r

def index_aux_y(model,i,f):
    """
    Index of the auxilliary variable that store the fulfilment of a specific need for a specific task
    """
    return (model.nb_resources()*model.nb_tasks())+(i*model.nb_features()+f)

def index_aux_z(model,i):
    """
    Index of the auxilliary variable that store the fulfulment of all needs for a specific task
    """
    return model.nb_tasks()*(model.nb_resources()+model.nb_features())+i


def x(model,i,r):
    return index_to_qubomodel(model,i,r)

def y(model,i,f):
    return index_aux_y(model,i,f)

def z(model,i):
    return index_aux_z(model,i)

def qubomodel_to_index(model,q):
    i = q//model.nb_resources()
    r = q%model.nb_resources()
    return (i,r)

def empty_qubo(model):
    return np.zeros((model.nb_tasks()*(model.nb_resources()+model.nb_features()+1),model.nb_tasks()*(model.nb_resources()+model.nb_features()+1)))

def add_quadratic_term(Q,index1,index2,coef):
    Q[index1,index2] += coef/2
    Q[index2,index1] += coef/2

def add_linear_term(Q, index, coef):
    Q[index,index] += coef

def schedule_qubo(model):
    """
    Add the penality if the same resource is affected to two tasks at the same schedule
    """
    Q = empty_qubo(model)
    penalty = 20
    for s in range(model.nb_schedules()):
        for r in range(model.nb_resources()):
            for i in range(model.nb_tasks()):
                for j in range(i+1,model.nb_tasks()):
                    Q[index_to_qubomodel(model,i,r),index_to_qubomodel(model,j,r)] += penalty*model.schedules[i,s]*model.schedules[j,s]
                    Q[index_to_qubomodel(model,j,r),index_to_qubomodel(model,i,r)] += penalty*model.schedules[i,s]*model.schedules[j,s]
    return Q

def auxilliary_constraints(model):
    """
    Generate the penalty terms for the z(i,f) auxilliary variables
    """
    Q = empty_qubo(model)
    penalty = 20
    # Constraints for the y auxilliary variables
    for i in range(model.nb_tasks()):
        for f in range(model.nb_features()):
            for r in range(model.nb_resources()):
                add_linear_term(Q,x(model,i,r),model.resources[r,f]*penalty)
                add_quadratic_term(Q, y(model,i,f), x(model,i,r),(-1)*model.resources[r,f]*penalty)

    # Constraints for the z auxilliary variables
    penalty = 20
    for i in range(model.nb_tasks()):
        for f in range(model.nb_features()):
            add_linear_term(Q, z(model,i), (2*model.needs[i,f]-1)*penalty)
            add_linear_term(Q, y(model,i,f), model.needs[i,f]*penalty)
            add_quadratic_term(Q, y(model,i,f), z(model,i), (-2)*model.needs[i,f]*penalty)

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
                add_linear_term(Q, index_to_qubomodel(model,i,r), (-1)*model.needs[i,f]*model.resources[r,f])
    return Q

def needs_bonus(model):
    """
    Add a bonus if all the needs of a task are fulfilled (z auxilliary variable)
    """
    Q = empty_qubo(model)
    bonus = 5
    for i in range(model.nb_tasks()):
        add_linear_term(Q,z(model,i),bonus)
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
                add_quadratic_term(Q, index_to_qubomodel(model,i,r1), index_to_qubomodel(model,i,r2), penalty)
    return Q

def generate_qubo(model):
    return schedule_qubo(model) + needs_bonus(model) + auxilliary_constraints(model)# +  single_assignement_qubo(model)
    #return schedule_qubo(model) + needs_bonus(model) + auxilliary_constraints(model) + single_assignement_qubo(model)

def qubo_solution_to_affectation_matrix(model,sample):
    return np.array(list(sample.values())[:model.nb_tasks()*model.nb_resources()]).reshape(model.nb_tasks(),model.nb_resources())

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
    return problemmodel.Solution(model,solution,"SimulatedAnnealing",duration)

def solve_on_dwave(model):
    Q = generate_qubo(model)
    bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
    start = time.time()

    sampler = EmbeddingComposite(DWaveSampler())   
    sampleset = sampler.sample(bqm,num_reads=100)

    print(sampleset.lowest(atol=.5))  
    duration = (time.time() - start)*1000
    qubo_solution = sampleset.first.sample
    solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
    return problemmodel.Solution(model,solution,"DWave QPU",duration)