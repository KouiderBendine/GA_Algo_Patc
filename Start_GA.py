import numpy as np
import matplotlib.pyplot as plt
import math
import subprocess
import csv
import os
import random
import time
import xlsxwriter

start_time = time.time()
num_generations = 400
Patch_Number = 8
Population_size = int(100)
qessi = 0.019
N_Mode = 6
mutation_rate = 7
X_vec = np.zeros(9)
Pos_Accum = np.zeros((Patch_Number, Population_size))
Vectors = np.zeros((Patch_Number, Population_size))
best_outputs = np.zeros((Population_size))
Pos_vect = np.zeros((Patch_Number, Population_size))

plt_gen = np.ceil(np.linspace(20, 200, 5))

def Obj_Fun(Pos_vect,N_Mode,qessi):

    file1 = open("Iter_param.inp", "w+")
    file1.write('x1=%e\n' % Pos_vect[0])
    file1.write('x2=%e\n' % Pos_vect[1])
    file1.write('x3=%e\n' % Pos_vect[2])
    file1.write('x4=%e\n' % Pos_vect[3])
    file1.write('x5=%e\n' % Pos_vect[4])
    file1.write('x6=%e\n' % Pos_vect[5])
    file1.write('x7=%e\n' % Pos_vect[6])
    file1.write('x8=%e\n' % Pos_vect[7])
    #file1.write('iter=%e\n' % k)
    file1.close()
    cmd = ["C:\Program Files\ANSYS Inc\\v201\\ansys\\bin\winx64\ANSYS201.exe", '-b', '-p', 'ANSYS', '-i',
           'D:\Vibration_control_Paper\Ansys_Code\GA_Algo_Patc\Code_AVC_ANN.txt', '-o',
           'D:\Vibration_control_Paper\Ansys_Code\GA_Algo_Patc\\ansys_out.txt']
    subprocess.call(cmd)

    ## Read The Output Results
    with open(os.path.join("D:\Vibration_control_Paper\Ansys_Code\GA_Algo_Patc","State_Space.csv"), 'r') as file:
        csv_reader = csv.reader(file, delimiter=",", quotechar='|')
        i = 0
        Total_vec = np.zeros((4, 12))
        for line in csv_reader:
            Total_vec[i, :] = np.matrix(line)
            i = i + 1

    B_vect = Total_vec[:, 0:6]
    A_vect = Total_vec[:, 6:12]

    ## Calculate The fitness function
    LamdaS = 0
    LamdaP = 1
    for N in range(N_Mode):

        LamdaS = LamdaS + np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[1, N])
        LamdaP = LamdaP * np.sum(np.square(B_vect[:, N])) / (4 * qessi * 2 * np.pi * A_vect[1, N])
    Fitn= LamdaS * (LamdaP) ** (1 / N_Mode)
    return Fitn


def crossover(parent_1, parent_2):
    # Get length of chromosome
    chromosome_length = len(parent_1)
    # Pick crossover point, avoding ends of chromsome
    crossover_point = random.randint(1, chromosome_length - 1)
    # Create children. np.hstack joins two arrays
    child_1 = np.hstack((parent_1[0:crossover_point],
                         parent_2[crossover_point:]))
    child_2 = np.hstack((parent_2[0:crossover_point],
                         parent_1[crossover_point:]))
    # Return children
    return child_1, child_2


def mutate(population, X_vec, mutation_rate):
    # Apply random mutation
    alf = int(mutation_rate*np.size(population)/100)
    #random_Row = np.random.choice(np.arange(1, population.shape[0]), n_mutations, replace=True)
    #random_Col = np.random.choice(population.shape[1], n_mutations, replace=True)

    for i in range(alf):
        random_Row = random.randint(0, np.size(population, axis=0)-1)
        random_Col = random.randint(0, np.size(population, axis=1)-1)
        population[random_Row, random_Col] = 1e-3 * np.random.choice(X_vec, 1)

    # Return mutation population
    return population



#Best_Position = np.zeros((Patch_Number, 1))
#Best_Fitness = np.zeros((num_generations))
n_mutations = math.ceil((Population_size - 1) * Patch_Number * mutation_rate)

## Creat the starting Population
for j in range(9):
    X_vec[j] = (j+1)*5+(j)*50
for it in range(Population_size):
    Pos_vect[:, it] = 1e-3 * np.random.choice(X_vec, Patch_Number)

## Fitness initial evaluation
for iter in range(Population_size):
    Ftn = Obj_Fun(Pos_vect[:, iter], N_Mode, qessi)
    best_outputs[iter] = Ftn
    Ranking = np.argsort(-1*best_outputs)
    Vectors = np.array(Pos_vect[:, Ranking])
best_outputs = best_outputs[Ranking]
best_outputs = best_outputs[0:int(Population_size/2)]

Best_Fitness = [best_outputs[0]]
Best_global = Best_Fitness[0]
Best_Position = [Vectors[:, 0]]
Vectors = Vectors[:, 0:int(Population_size/2)]


##--------------------------------------
# Position
A = np.array(Vectors)
Length = np.linspace(0, 0.5, 10)
Width = np.linspace(0, 0.5, 10)
x_L, y_w = np.meshgrid(Length, Width)

[p, h] = A.shape

# random_matrix = np.random.rand(p, h) * 0.02
random_matrix = np.random.uniform(low=0, high=0.04, size=(p, h))
alfa = np.zeros((p, h))
for t in range(h):
    for itr in range(p):
        alfa[itr, t] = A[itr, t] + random_matrix[itr, t]

# for i in range(h - 4):
# plt.clf()
plt.figure(1)
# xx, yy = np.meshgrid(alfa[:,], alfa)

for pl in range(4):
    ax1 = plt.plot(alfa[:, 2 * pl], alfa[:, 2 * pl + 1], color='r', marker='*', linestyle='none')
ax2 = plt.plot(x_L, y_w, color='k', linewidth=3.0)
ax3 = plt.plot(y_w, x_L, color='k', linewidth=3.0)
plt.show()
#score = [Best_Fitness]
print('Starting best score, percent target: %.10f' %Best_global)
for generation in range(num_generations):
    print("Generation : ", generation, "Best Fitness :", Best_Fitness[-1], "Best Fitness :", Best_Position[-1])


    # Create an empty list for new population

    new_population = []

    # Create new popualtion generating two children at a time

    for J in range(int(Population_size/4)):

         parent_1 = Vectors[:, 2*J]
         parent_2 = Vectors[:, 2*J+1]
         child_1, child_2 = crossover(parent_1, parent_2)
         new_population.append(child_1)
         new_population.append(child_2)


    # Replace the old population with the new one

    population = np.transpose(np.array(new_population))

    # Mutate the population

    population = mutate(population, X_vec, n_mutations)

    # Evaluate the fitness function

    for iter in range(int(Population_size/2)):
        Ftn = Obj_Fun(population[:, iter], N_Mode, qessi)
        best_outputs[iter] = Ftn
        Ranking = np.argsort(-1*best_outputs)
        Vectors = np.array(population[:, Ranking])
    best_outputs = best_outputs[Ranking]
    Best_local = best_outputs[0]

    if Best_local >= Best_global:
       Best_Position.append(Vectors[:, 0])
       Best_Fitness.append(Best_local)
       Best_global = Best_local
    else:
       Best_Position.append(Best_Position[generation])
       Best_Fitness.append(Best_Fitness[generation])




# Plot the results
plt.figure(2)
plt.plot(Best_Fitness, marker='o')
plt.xlabel('Generation', fontsize=18)
plt.ylabel('Fitness', fontsize=18)

plt.grid()
plt.show()
print(Best_Fitness)
print('**************')
print(Best_Position[-1])

# Position
A = np.array(Best_Position)
Length = np.linspace(0, 0.5, 10)
Width = np.linspace(0, 0.5, 10)
x_L, y_w = np.meshgrid(Length, Width)

[p, h] = A.shape

# random_matrix = np.random.rand(p, h) * 0.02
random_matrix = np.random.uniform(low=0, high=0.04, size=(p, h))
alfa = np.zeros((p, h))
for t in range(h):
    for itr in range(p):
        alfa[itr, t] = A[itr, t] + random_matrix[itr, t]

plt.figure(3)
for pl in range(4):
    ax1 = plt.plot(alfa[:, 2 * pl], alfa[:, 2 * pl + 1], color='r', marker='*', linestyle='none')
ax2 = plt.plot(x_L, y_w, color='k', linewidth=3.0)
ax3 = plt.plot(y_w, x_L, color='k', linewidth=3.0)
plt.show()

# Save the best results
workbook = xlsxwriter.Workbook('Opt_Posin.xlsx')

worksheet = workbook.add_worksheet()
row = 0
for col, data in enumerate(Best_Position):
    worksheet.write_column(row, col, data)

Rw = Patch_Number+1
worksheet.write_row(Rw, 0, np.transpose(np.array(Best_Fitness)))

workbook.close()

print(time.time() - start_time, "seconds")






