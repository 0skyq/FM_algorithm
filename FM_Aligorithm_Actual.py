import numpy as np
import pandas as pd

t = int(input("Enter the number of Tasks: "))
p = int(input("Enter the number of Processors: "))
D = int(input("Enter Time constraint value:"))

E = np.random.randint(5, 20, size=(t, p))
T = np.random.randint(20, 50, size=(t, p))
C = np.random.randint(5, 20, size=(t, t))
C = (C+C.T)
np.fill_diagonal(C,0)

"""
E = np.zeros((t, p))
T = np.zeros((t, p))
C = np.zeros((t, t))

print("\n")
print("Enter Energy Cost E[Task][Processor]:")
for i in range(t):
    for j in range(p):
        value = int(input(f"E[{i}][{j}]: "))
        E[i][j] = value

print("Enter Execution Time T[Task][Processor]:")
for i in range(t):
    for j in range(p):
        value = int(input(f"T[{i}][{j}]: "))
        T[i][j] = value

print("Communication costs C[Task-i][Task-j]:")
for i in range(t):
    for j in range(i, t):
        if i != j:
           value = float(input(f"C[{i}][{j}]: "))
           C[i][j] = value
           C[j][i] = value
        if i == j:
            C[i][j] = 0
"""
print("Energy costs:")
for row in E:
    print(row)
print("Execution Times:")
for row in T:
    print(row)
print("Communication costs:")
for row in C:
    print(row)


def Gain_cal(test_array, t, moved_task, x):
    gain = np.zeros((p, t))
    stop = 0
    for maybe in range(p):
        for m in range(t):
            C_cost = sum(C[m][j] for j in range(t) if test_array[m] != test_array[j])
            C_cost_maybe = sum(C[m][j] for j in range(t) if maybe != test_array[j])

            now_cost = E[m][test_array[m]] + C_cost
            maybe_cost = E[m][maybe] + C_cost_maybe
            gain[maybe][m] = now_cost - maybe_cost
            if maybe == test_array[m]:
                gain[maybe][m] = None

    if moved_task[0] != None:
        for i in range(t):
            if moved_task[i] != None:
                gain[:, moved_task[i]] = None

    print("\ngain array:\n", gain)

    max_gain = np.nanmax(gain)

    if (x > 0):
        next_max_gain = gain.flatten()
        next_max_gain = np.nan_to_num(next_max_gain,nan=-1000)
        indices = np.where(next_max_gain == -1000)
        next_max_gain = np.delete(next_max_gain, indices)
        s_array = np.sort(next_max_gain)[::-1]
        #print(s_array)
        if x < len(s_array):
            max_gain = s_array[x]

        if x > len(s_array):
            stop = 1


    which_partition,which_task = np.where(gain == max_gain)

    return max_gain, which_task[0], which_partition[0], stop
def Constraint(temp_array):
    print("\nCHECKING CONSTRAINT\n")
    z = 1
    x = np.zeros(p)
    APT = np.zeros(p)
    y = np.zeros(p)

    for i in range(p):
        for j in range(t):
            if i == temp_array[j]:
                x[i] += T[j][temp_array[j]]
        APT[i] = x[i]

    for i in range(p):
        if i < p - 1:
            y[i] = abs(APT[i] - APT[i + 1])
        else:
            y[i] = abs(APT[p - 1] - APT[0])

    for i in range(p):
        if y[i] > D:
            z = 0

    print("Aggregate Execution Times are:", APT)
    print("Their Differences are:", y)

    return z,APT
def Update():
    Initial_Array = np.random.randint(0, p, t)
    A_Array = np.copy(Initial_Array)
    Final_Array = np.copy(Initial_Array)
    moved_task = [None] * t
    Total_cost = np.zeros(t)

    print("\nROWS ARE PROCESSORS (P0,P1,P2...)")
    print("COLUMNS ARE TASKS (T0,T1,T2...")
    print("\nINITIAL SOLUTION :", Initial_Array)

    C_cost = 0
    Initial_cost = 0
    for m in range(t):
        C_cost = sum(C[m][j] for j in range(t) if Initial_Array[m] != Initial_Array[j])
        C_cost = C_cost / 2
        Initial_cost += E[m][Initial_Array[m]] + C_cost
    print(f"Initial Energy COST  is: {Initial_cost}")

    z,APT=Constraint(Initial_Array)
    #print("Initial Aggregate Execution Times are:", APT)

    for i in range(t):
        test_array = np.copy(A_Array)
        x = 0
        max_gain, which_task, which_partition, stop = Gain_cal(test_array, t, moved_task, x)
        print(f"max_gain={max_gain}")
        print(f"Temporarily moving Task {which_task} to Partition {which_partition} to check Constraint")
        test_array[which_task] = which_partition

        z,APT = Constraint(test_array)
       # print(f"z = {z}")

        while z == 0 and stop != 1:
            print("Constraint not bounded")
            print("choose next best gain, loop:", x)
            test_array = np.copy(A_Array)
            x = x + 1
            max_gain, which_task, which_partition, stop = Gain_cal(test_array, t, moved_task, x)
            print(f"Temporarily moving Task {which_task} to Partition {which_partition} to check Constraint")
            print(f"next_max_gain={max_gain}")
            test_array[which_task] = which_partition
            z ,APT= Constraint(test_array)

        if stop == 1:
            x = 0
            print("NO POSSIBLE MOVES")
            z = 2

        if z == 1:
            x = 0
            stop = 0
            print("Constraint bounded")
            print(f"Task {which_task} is moved  to Partition {which_partition}")
            moved_task[i] = which_task
            A_Array[which_task] = which_partition
           # print("moved_task:", moved_task)
           # print("A_Array:", A_Array)
            C_cost = 0
            Total_cost[i] = 0

            for m in range(t):
                C_cost = sum(C[m][j] for j in range(t) if A_Array[m] != A_Array[j])
                C_cost = C_cost / 2
                Total_cost[i] += E[m][A_Array[m]] + C_cost

    # print(f"Energy COST after moving task {which_task} is: {Total_cost[i]}")
    Total_cost = np.where(Total_cost == 0, np.nan, Total_cost)

    T = np.nanmin(Total_cost)
    #print(T)
    k = 0
    for i in range(t):
        if T == Total_cost[i]:
            k = i

    for m in range(t):
        for n in range(k+1):
            if moved_task[n] == m:
                Final_Array[m] = A_Array[m]
    C_cost = 0
    Final_cost = 0
    for m in range(t):
        C_cost = sum(C[m][j] for j in range(t) if Final_Array[m] != Final_Array[j])
        C_cost = C_cost / 2
        Final_cost += E[m][Final_Array[m]] + C_cost

    if Final_cost > Initial_cost:
        Final_Array = np.copy(Initial_Array)
        Final_cost = Initial_cost

    z,APT=Constraint(Final_Array)

    df = pd.DataFrame(Final_Array, columns=["        "])
    print("Task    Partition")
    print(df)

    print("\n   -------    FINAL RESULTS   --------\n")
    print("TASK MOVEMENT:", moved_task)
    print("ENERGY COST AFTER EACH MOVEMENT:", Total_cost)
    #print(f"OPTIMAL COST at k={k+1} and Respective Total_Cost={Total_cost[k]}")
    print("FINAL SOLUTION", Final_Array)
    print(f"Final Energy COST  is: {Final_cost}")
    #print("Final Aggregate Execution Times are:", APT)


w=1
if w==1:
    Update()