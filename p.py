import pulp

tasks = [
    ("R.01", 80, 9, 1.3, 0.0, [], []),
    ("R.02", 85, 8, 1.7, 0.1, ["R.03"], []),
    ("R.05", 85, 9, 1.3, 0.3, ["R.02"], []),
    ("R.06", 60, 5, 1.0, 0.3, [], ["R.01", "R.02"]),
    ("R.07", 50, 6, 1.0, 0.0, [], ["R.09", "R.10"]),
    ("R.08", 40, 4, 1.0, 0.0, [], ["R.01"]),
    ("R.09", 50, 6, 1.0, 0.0, [], []),
    ("R.10", 40, 3, 1.0, 0.0, [], ["R.09"]),
    ("R.13", 70, 9, 1.3, 0.3, ["R.14"], []),
    ("R.14", 75, 7, 1.3, 0.3, ["R.13"], []),
    ("R.15", 60, 7, 1.0, 0.0, [], []),
    ("R.16", 55, 7, 1.0, 0.0, [], []),
    ("R.17", 20, 4, 1.7, 0.2, ["R.18", "R.19"], ["R.13", "R.14"]),
    ("R.18", 25, 4, 1.0, 0.0, [], ["R.17"]),
    ("R.19", 25, 5, 1.0, 0.0, [], ["R.13", "R.14"]),
    ("R.20", 15, 6, 1.0, 0.0, [], []),
    ("R.53", 80, 7, 1.3, 0.0, [], []),
    ("R.54", 55, 7, 1.0, 0.0, [], []),
    ("R.55", 50, 4, 1.0, 0.0, [], [])
]

Sprints = range(5)
Capacity = [20, 20, 35, 30, 20]

UserStories = list(range(len(tasks)))
task_id = {j: tasks[j][0] for j in UserStories}
Uj = {j: tasks[j][1] for j in UserStories}
Pj = {j: tasks[j][2] for j in UserStories}
rj = {j: tasks[j][3] for j in UserStories}
aj = {j: tasks[j][4] for j in UserStories}
Yj = {j: [i for i, t in enumerate(tasks) if t[0] in tasks[j][5]] for j in UserStories if tasks[j][4] > 0}
Dj = {j: [i for i, t in enumerate(tasks) if t[0] in tasks[j][6]] for j in UserStories if tasks[j][6]}

model = pulp.LpProblem("Sprint_Planning", pulp.LpMaximize)

# Variables
X = pulp.LpVariable.dicts("X", ((i, j) for i in Sprints for j in UserStories), cat="Binary")
Y = pulp.LpVariable.dicts("Y", ((i, j) for i in Sprints for j in Yj), lowBound=0, cat="Integer")

# Main function
model += (
    pulp.lpSum(Uj[j] * rj[j] * X[i, j] for i in Sprints for j in UserStories)# №+
    #pulp.lpSum(Uj[j] * aj[j] * Y[i, j] / len(Yj[j]) for i in Sprints for j in Yj if len(Yj[j]) > 0)
)

# Limits
for i in Sprints:
    model += pulp.lpSum(Pj[j] * X[i, j] for j in UserStories) <= Capacity[i]

# Once
for j in UserStories:
    model += pulp.lpSum(X[i, j] for i in Sprints) <= 1

# Corr
for i in Sprints:
    for j in Yj:
        model += Y[i, j] <= pulp.lpSum(X[i, k] for k in Yj[j])
        model += Y[i, j] <= len(Yj[j]) * X[i, j]

# Depends
for j, preds in Dj.items():
    for p in preds:
        for t in Sprints:
            model += pulp.lpSum(X[k, p] for k in range(0, t + 1)) >= X[t, j]

model.solve()

print("Status:", pulp.LpStatus[model.status])
print("Score:", pulp.value(model.objective))

for i in Sprints:
    print(f"\nSprint {i+1} (max. {Capacity[i]} SP):")
    used = 0
    for j in UserStories:
        if pulp.value(X[i, j]) == 1:
            sp = Pj[j]
            used += sp
            print(f"--- {task_id[j]} (Uj={Uj[j]}, SP={sp})")
    print(f"  Used: {used}/{Capacity[i]}")