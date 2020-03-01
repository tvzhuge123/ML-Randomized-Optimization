import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import sklearn
import mlrose

random_states = [1,2,3,4,5,6,7,8,9,10]
size = [10,20,30,40,50,60,70,80,90,100]
algos = ['RHC', 'SA', 'GA', 'MIMIC']

def validate(fitness, size, name):
    if name=='Knapsack': size = [20]

    df_optim = pd.DataFrame(columns=algos)
    df_iter = pd.DataFrame(columns=algos)
    df_time = pd.DataFrame(columns=algos)
    for s in size:
        print(s)
        problem = mlrose.DiscreteOpt(length=s, fitness_fn=fitness, maximize=True, max_val=2)
        df_optim.loc[len(df_optim)], df_iter.loc[len(df_iter)], df_time.loc[len(df_time)] = optimization(problem)

    df_optim.set_index(pd.Index(size), inplace=True)
    df_iter.set_index(pd.Index(size), inplace=True)
    df_time.set_index(pd.Index(size), inplace=True)

    df_optim.to_csv(name + '_optim.csv')
    df_iter.to_csv(name + '_iter.csv')
    df_time.to_csv(name + '_time.csv')

    OptPlot(df_optim, df_iter, df_time)


def optimization(problem):
    df_optim = pd.DataFrame(columns=algos)
    df_iter = pd.DataFrame(columns=algos)
    df_time = pd.DataFrame(columns=algos)

    for rs in random_states:
        # RHC
        tic = time.process_time()
        best_state_rhc, best_fitness_rhc, curve_rhc = \
            mlrose.random_hill_climb(problem, max_attempts=20, max_iters=100000,
                                     restarts=0, init_state=None, curve=True, random_state=rs)
        toc = time.process_time()
        time_rhc = toc - tic

        # SA
        tic = time.process_time()
        best_state_sa, best_fitness_sa, curve_sa = \
            mlrose.simulated_annealing(problem, schedule=mlrose.ExpDecay(init_temp=1.0, exp_const=0.005, min_temp=0.001),
                                       max_attempts = 20, max_iters = 100000,
                                       init_state = None, curve = True, random_state = rs)
        toc = time.process_time()
        time_sa = toc - tic

        # GA
        tic = time.process_time()
        best_state_ga, best_fitness_ga, curve_ga = \
            mlrose.genetic_alg(problem, pop_size=200, mutation_prob=0.1, max_attempts=20, max_iters=100000,
                               curve=True, random_state=rs)
        toc = time.process_time()
        time_ga = toc - tic

        # MIMIC
        tic = time.process_time()
        best_state_m, best_fitness_m, curve_m = \
            mlrose.mimic(problem, pop_size=20, keep_pct=0.2, max_attempts=20, max_iters=100000,
                         curve=True, random_state=rs, fast_mimic=False)
        toc = time.process_time()
        time_m = toc - tic

        # df
        df_optim.loc[len(df_optim)] = [best_fitness_rhc, best_fitness_sa, best_fitness_ga, best_fitness_m]
        df_iter.loc[len(df_iter)] = [len(curve_rhc), len(curve_sa), len(curve_ga), len(curve_m)]
        df_time.loc[len(df_time)] = [time_rhc, time_sa, time_ga, time_m]
        print(rs)

    return(df_optim.mean(axis=0), df_iter.mean(axis=0), df_time.mean(axis=0))


def OptPlot(df_optim, df_iter, df_time):
    fig = plt.figure(figsize=(10,10))

    ax_optim = fig.add_subplot(3, 1, 1)
    ax_optim.set_ylabel("Optimal Solution")

    ax_iter = fig.add_subplot(3, 1, 2)
    ax_iter.set_ylabel("Number of Iterations")

    ax_time = fig.add_subplot(3, 1, 3)
    ax_time.set_xlabel("Array Size N")
    ax_time.set_ylabel("Process Time (s)")

    df_optim.plot(grid=True, marker='o', xticks=df_optim.index, title='Optimal Solution vs Input Size', ax=ax_optim)
    df_iter.plot(grid=True, marker='o', xticks=df_iter.index, title='Number of Iterations vs Input Size', ax=ax_iter)
    df_time.plot(grid=True, marker='o', xticks=df_time.index, title='Process Time (s) vs Input Size', ax=ax_time)

    plt.show()


def OptCounOnes():
    fitness = mlrose.OneMax()
    validate(fitness, size, 'CounOnes')


def OptFourPeaks():
    fitness = mlrose.FourPeaks(t_pct=0.1)
    validate(fitness, size, 'FourPeaks')


def OptKnapsack():
    fitness = mlrose.Knapsack(weights=[10, 5, 2, 8, 15, 4, 3, 6, 7, 20, 21, 25, 22, 28, 25, 24, 23, 26, 27, 30],
                              values=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20],
                              max_weight_pct=0.6)
    validate(fitness, size, 'Knapsack')


def main():
    OptCounOnes()
    OptFourPeaks()
    OptKnapsack()


if __name__ == '__main__':
    main()