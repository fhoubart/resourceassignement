from problemmodel import *
from qubo import *
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import RandomSampler
from dimod import ExactSolver
import dimod
from model import *
from qubo import *
from dimod.serialization.format import Formatter



model = small_sample_problem()
print(model.task_features(1))
print(model.resources.shape)
print(model.needs.shape)
print(model.schedules.shape)

Q = generate_qubo(model)
bqm = dimod.BinaryQuadraticModel.from_numpy_matrix(Q)
sampler = SimulatedAnnealingSampler()
#    sampleset = sampler.sample(bqm,num_reads=1000, beta_range=[.1, 4.2], beta_schedule_type='linear',initial_state_generator="random")
sampleset = sampler.sample(bqm,num_reads=1000, beta_schedule_type='linear', initial_state_generator="random")
#sampleset = ExactSolver().sample(bqm)
#sampleset = RandomSampler().sample(bqm,num_reads=200000)
print(bqm.quadratic)
print(bqm.linear)
print(sum(bqm.linear))
#exit()
Formatter(depth=10).fprint(sampleset)

print('Qubo solution:')
qubo_solution = sampleset.first.sample
print(qubo_solution)
solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
print("Solution matrix:")
print(solution)

print("")
print("Chacking validity...")
model.check_solution(solution)
print("")
print("Coverage is ",model.coverage(solution))
