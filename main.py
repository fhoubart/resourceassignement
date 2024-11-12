from problemmodel import *
from qubo import *
from dwave.samplers import SimulatedAnnealingSampler
from dwave.samplers import RandomSampler
from dimod import ExactSolver
import dimod
from qubo import *
from dimod.serialization.format import Formatter


def energy(Q,solution):
    e = 0
    (n,m) = Q.shape
    for i in range(n):
        for j in range(m):
            e += Q[i,j]*solution[i]*solution[j]
    return e

def test_model():
    model = super_easy_sample_problem()
    Q = generate_qubo(model)

    solutions = [
        # perfect solution
        [1,0,0,0,   0,1,0,0,   0,0,0,1,    1,1,1,  1,1,1,  1,1,1,   1,1,1  ],
        # Uncomplete solution
        [1,0,0,0,   0,1,0,0,   0,0,1,0,    1,1,1,  1,1,1,  0,1,1,   1,1,0  ],
        # Error in auxilliary z
        [1,0,0,0,   0,1,0,0,   0,0,0,1,    1,1,1,  1,1,1,  1,1,1,   0,1,1  ],
        # Error in auxilliary y
        [1,0,0,0,   0,1,0,0,   0,0,0,1,    1,1,1,  1,1,1,  1,1,0,   1,1,1  ],
        # Duplicate affectation
        [1,0,0,0,   0,1,0,0,   1,0,0,1,    1,1,1,  1,1,1,  1,1,1,   1,1,1  ],
        # Uncomplete solution with error on y to make it seems complete ERROR
        [1,0,0,0,   0,1,0,0,   0,0,1,0,    1,1,1,  1,1,1,  1,1,1,   1,1,1  ],
        # Uncomplete solution with error on z to make it seems complete
        [1,0,0,0,   0,1,0,0,   0,0,1,0,    1,1,1,  1,1,1,  0,1,1,   1,1,1  ],

    ]

    for i,s in enumerate(solutions):
        qubo_solution = {}
        for index,value in enumerate(s):
            qubo_solution[index] = value
        solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
        print(f"{i}: energy: {energy(Q,s)}, coverage: {model.coverage(solution)}")

test_model()
exit()

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

print()
print("Problem size: ")
print(f"  - tasks: {model.nb_tasks()}")
print(f"  - resources: {model.nb_resources()}")
print(f"  - schedules: {model.nb_schedules()}")
print(f"  - features: {model.nb_features()}")
print('Qubo solution:')
num=0
for data in sampleset.data():
    qubo_solution = data.sample
    solution = qubo_solution_to_affectation_matrix(model,qubo_solution)
    print(f"- energy: {data.energy}, coverage: {model.coverage(solution)}")
    if num > 50:
        break
    num += 1

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
