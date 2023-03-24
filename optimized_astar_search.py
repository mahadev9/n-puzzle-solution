import queue
import ujson
from pprof import cpu
from itertools import count
from heapq import heappush, heappop


class State:
    def __init__(self, initial_state, depth=0, previous_state=None, action=None, cost=0):
        self.state = initial_state
        self.depth = depth
        self.previous_state = previous_state
        self.action = action
        self.cost = cost


@cpu
def possible_actions(initial_state):
    actions = []
    current_state = initial_state.state
    for i in range(len(current_state)):
        for j in range(len(current_state[i])):
            if current_state[i][j] is None:
                if i-1 >= 0:
                    actions.append([UP, i-1, j])
                if i+1 < len(current_state):
                    actions.append([DOWN, i+1, j])
                if j-1 >= 0:
                    actions.append([LEFT, i, j-1])
                if j+1 < len(current_state[i]):
                    actions.append([RIGHT, i, j+1])
                break
    return actions


@cpu
def result(action, initial_state):
    if action[0] == UP:
        initial_state[action[1]][action[2]], initial_state[action[1]+1][action[2]] = \
            initial_state[action[1]+1][action[2]], initial_state[action[1]][action[2]]
    elif action[0] == DOWN:
        initial_state[action[1]][action[2]], initial_state[action[1]-1][action[2]] = \
            initial_state[action[1]-1][action[2]], initial_state[action[1]][action[2]]
    elif action[0] == LEFT:
        initial_state[action[1]][action[2]], initial_state[action[1]][action[2]+1] = \
            initial_state[action[1]][action[2]+1], initial_state[action[1]][action[2]]
    elif action[0] == RIGHT:
        initial_state[action[1]][action[2]], initial_state[action[1]][action[2]-1] = \
            initial_state[action[1]][action[2]-1], initial_state[action[1]][action[2]]
    return initial_state


@cpu
def expand(initial_state):
    actions = possible_actions(initial_state)
    states = []
    for action in actions:
        intermediateState = result(action, ujson.loads(ujson.dumps(initial_state.state)))
        state = State(intermediateState, initial_state.depth+1, initial_state, action, initial_state.cost+1)
        states.append(state)
    return states


@cpu
def a_star_search(initial_state, goal_state, heuristic_fun):
    if initial_state.state == goal_state:
        return initial_state, 0
    c = count()
    qStates = [(0, next(c), initial_state.cost, str(initial_state.state), initial_state)]
    closed = {}
    opened = {}
    max_queue_size = 1
    while qStates:
        if max_queue_size <= len(qStates):
            max_queue_size = len(qStates)
        _, _, currentCost, _, currentState = heappop(qStates)
        if currentState.state == goal_state:
            return currentState, max_queue_size
        if str(currentState.state) in closed:
            continue
        closed[str(currentState.state)] = currentState
        newCurrentCost = currentCost + 1
        childStates = expand(currentState)
        for childState in childStates:
            if str(childState.state) in closed:
                continue
            if str(childState.state) in opened:
                childCost, childHeuristic = opened[str(childState.state)]
                if newCurrentCost >= childCost:
                    continue
            else:
                childHeuristic = heuristic_fun(childState.state, goal_state)
            opened[str(childState.state)] = newCurrentCost, childHeuristic
            heappush(qStates, (childHeuristic + newCurrentCost, next(c), newCurrentCost, str(childState.state), childState))
    return None, max_queue_size


@cpu
def manhattan_heuristic(current_state, goal_state, size):
    distance = 0
    for k in range(len(current_state)):
        for l in range(len(current_state[k])):
            if current_state[k][l] != None and current_state[k][l] != goal_state[k][l]:
                goal_index = current_state[k][l] if current_state[k][l] != None else 0
                distance += abs(goal_index//size - k) + abs(goal_index%size - l)
    return distance


@cpu
def number_of_conflicts(current_state_row, goal_state_row, size, nConflicts=0):
    conflicts = [0 for x in range(size)]
    for i, tile_1 in enumerate(current_state_row):
        if tile_1 in goal_state_row and tile_1 != None:
            goal_state_i = goal_state_row.index(tile_1)
            for j, tile_2 in enumerate(current_state_row):
                if tile_2 in goal_state_row and tile_2 != None and i != j:
                    goal_state_j = goal_state_row.index(tile_2)
                    if goal_state_i > goal_state_j and i < j:
                        conflicts[i] += 1
                    if goal_state_i < goal_state_j and i > j:
                        conflicts[i] += 1
    if max(conflicts) == 0:
        return nConflicts * 2
    else:
        i = conflicts.index(max(conflicts))
        current_state_row[i] = -1
        nConflicts += 1
        return number_of_conflicts(current_state_row, goal_state_row, size, nConflicts)


@cpu
def linear_conflicts(current_state, goal_state, size=4):
    conflicts = manhattan_heuristic(current_state, goal_state, size)
    current_state_rows = []
    current_state_cols = []
    goal_state_rows = []
    goal_state_cols = []
    for row in range(len(goal_state)):
        c_row = []
        c_col = []
        g_row = []
        g_col = []
        for col in range(len(goal_state[row])):
            c_row.append(current_state[row][col])
            c_col.append(current_state[col][row])
            g_row.append(goal_state[row][col])
            g_col.append(goal_state[col][row])
        current_state_rows.append(c_row)
        current_state_cols.append(c_col)
        goal_state_rows.append(g_row)
        goal_state_cols.append(g_col)
    for i in range(size):
        conflicts += number_of_conflicts(current_state_rows[i], goal_state_rows[i], size)
    for i in range(size):
        conflicts += number_of_conflicts(current_state_cols[i], goal_state_cols[i], size)
    return conflicts


@cpu
def show_solution(final_state):
    actions = []
    states = []
    while final_state.previous_state is not None:
        actions.insert(0, final_state.action)
        states.insert(0, final_state.state)
        final_state = final_state.previous_state
    print("Sequence of actions: ", *actions, sep='\n')
    print("Sequence of states: ", *states, sep='\n')
    print("Path Cost: ", len(actions))


if __name__ == "__main__":
    cpu.auto_report()

    puzzle6 = [[8, 7, 6], [5, 4, 3], [2, 1, None]]
    goalState8 = [[None, 1, 2], [3, 4, 5], [6, 7, 8]]

    puzzle8 = [[13, 10, 11, 6], [5, 7, 4, 8], [1, 12, 14, 9], [3, 15, 2, None]]
    puzzle9 = [[13, 10, 11, 6], [5, 7, 4, 8], [2, 12, 14, 9], [3, 15, 1, None]]

    goalState15 = [[None, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    initial_state = State(puzzle8)
    goalState = goalState15

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"
    FAILURE = "FAILURE"
    CUTOFF = "CUTOFF"

    finalState, max_queue_size = a_star_search(initial_state, goalState, linear_conflicts)

    print("Initial state: ", initial_state.state)
    show_solution(finalState)
    print("Max Queue Size: ", max_queue_size)
