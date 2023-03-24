import queue
import ujson
from pprof import cpu


class State:
    def __init__(self, initial_state, depth=0, previous_state=None, action=None, cost=0):
        self.state = initial_state
        self.depth = depth
        self.previous_state = previous_state
        self.action = action
        self.cost = cost


class CustomPriorityQueue:
    def __init__(self):
        self.queue = {}
        self.length = 0

    def empty(self):
        if self.length == 0:
            return True
        return False

    def put(self, item):
        priority, state = item
        self.length += 1
        if priority in self.queue:
            self.queue[priority].append(item)
        else:
            self.queue[priority] = [item]
        # index = len(self.queue)
        # for idx, (p, s) in enumerate(self.queue):
        #     if priority < p:
        #         index = idx
        #         break
        # self.queue.insert(index, item)

    def get(self):
        element = "Queue is empty"
        min_key = min(self.queue.keys())
        element = self.queue[min_key].pop(0)
        self.length -= 1
        if len(self.queue[min_key]) < 1:
            del self.queue[min_key]
        return element

    def qsize(self):
        return self.length
    
    def print_queue(self):
        for p in self.queue:
            for item in self.queue[p]:
                print(item)


@cpu
def get_inversion_count(puzzle):
    count = 0
    x_pos = -1
    linear = []
    N = len(puzzle)
    for i in range(N):
        for itm in puzzle[i]:
            linear.append(itm)
            if itm == None:
                x_pos = i+1
    for i in range(N * N - 1):
        for j in range(i + 1, N * N):
              if linear[i] and linear[j] and linear[i] > linear[j]:
                  count += 1
    return count, x_pos


@cpu
def is_solvable(puzzle):
    inversionCount, blankPos = get_inversion_count(puzzle)
    if len(puzzle) % 2:
        return not bool(inversionCount % 2)
    else:
        if blankPos % 2:
            return not bool(inversionCount % 2)
        else:
            return bool(inversionCount % 2)


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
def is_cyclic(intermediateState):
    state = intermediateState.state
    while intermediateState.previous_state is not None:
        if state == intermediateState.previous_state.state:
            return True
        intermediateState = intermediateState.previous_state
    return False


@cpu
def iterate_deepening_search(initial_state, goal_state):
    for depth in range(50):
        # print("depth: ", depth)
        result, max_queue_size = depth_limited_search(initial_state, goal_state, depth)
        if result is not CUTOFF:
            return result, max_queue_size


@cpu
def depth_limited_search(initial_state, goal_state, depth):
    qStates = queue.LifoQueue()
    qStates.put(initial_state)
    result = FAILURE
    max_queue_size = 1
    while (not qStates.empty()):
        if max_queue_size < qStates.qsize():
            max_queue_size = qStates.qsize()
        currentState = qStates.get()
        if currentState.state == goal_state:
            return currentState, max_queue_size
        if currentState.depth > depth:
            result = CUTOFF
        elif not is_cyclic(currentState):
            states = expand(currentState)
            for state in states:
                qStates.put(state)
    return result, max_queue_size


@cpu
def bfs_search(initial_state, goal_state):
    if initial_state.state == goal_state:
        return initial_state, 0
    qStates = queue.Queue()
    qStates.put(initial_state)
    reached = {str(initial_state.state): initial_state}
    max_queue_size = 1
    while (not qStates.empty()):
        if max_queue_size < qStates.qsize():
            max_queue_size = qStates.qsize()
        currentState = qStates.get()
        if currentState.state == goal_state:
            return currentState, max_queue_size
        childStates = expand(currentState)
        for childState in childStates:
            if str(childState.state) not in reached:
                reached[str(childState.state)] = childState
                qStates.put(childState)
    return None, max_queue_size


@cpu
def a_star_search(initial_state, goal_state, heuristic_fun):
    if initial_state.state == goal_state:
        return initial_state, 0
    qStates = CustomPriorityQueue()
    initialCost = heuristic_fun(initial_state.state, goal_state)
    qStates.put((initialCost, initial_state))
    reached = {str(initial_state.state): initial_state}
    max_queue_size = 1
    while (not qStates.empty()):
        if max_queue_size < qStates.qsize():
            max_queue_size = qStates.qsize()
        currentCost, currentState = qStates.get()
        if currentState.state == goal_state:
            return currentState, max_queue_size
        childStates = expand(currentState)
        for childState in childStates:
            if str(childState.state) not in reached or childState.cost < reached[str(childState.state)].cost:
                h = heuristic_fun(childState.state, goal_state) + childState.cost
                reached[str(childState.state)] = childState
                qStates.put((h, childState))
    return None, max_queue_size


@cpu
def uniform_heuristic(current_state, goal_state):
    return 0


@cpu
def misplaced_heuristic(current_state, goal_state):
    misplaced_tiles = 0
    for row in range(len(current_state)):
        for col in range(len(current_state[row])):
            if current_state[row][col] != None and current_state[row][col] != goal_state[row][col]:
                misplaced_tiles += 1
    return misplaced_tiles


@cpu
def manhattan_heuristic(current_state, goal_state):
    distance = 0
    indexing = {}
    for i in range(len(goal_state)):
        for j in range(len(goal_state[i])):
            indexing[goal_state[i][j]] = [i, j]
    for k in range(len(current_state)):
        for l in range(len(current_state[k])):
            if current_state[k][l] != None and current_state[k][l] != goal_state[k][l]:
                distance += abs(indexing[current_state[k][l]][0] - k) + abs(indexing[current_state[k][l]][1] - l)
    return distance


@cpu
def linear_conflict_heuristic(current_state, goal_state):
    conflicts = 0
    goal_rows = []
    goal_cols = []
    for row in range(len(goal_state)):
        rows = []
        cols = []
        for col in range(len(goal_state[row])):
            rows.append(goal_state[row][col])
            cols.append(goal_state[col][row])
        goal_rows.append(rows)
        goal_cols.append(cols)

    for row in range(len(current_state)):
        for col in range(len(current_state[row])-1):
            for mid in range(col+1, len(current_state[row])):
                if current_state[row][col] != None \
                    and current_state[row][mid] != None \
                    and ((current_state[row][col] != goal_state[row][col]) or (current_state[row][mid] != goal_state[row][mid])) \
                    and current_state[row][col] in goal_rows[row] \
                    and current_state[row][mid] in goal_rows[row]:
                    conflicts += 2
                if current_state[col][row] != None \
                    and current_state[mid][row] != None \
                    and ((current_state[col][row] != goal_state[col][row]) or (current_state[mid][row] != goal_state[mid][row])) \
                    and current_state[col][row] in goal_cols[row] \
                    and current_state[mid][row] in goal_cols[row]:
                    conflicts += 2

    # for row in range(len(current_state)):
    #     for col in range(len(current_state[row])-1):
    #         for mid in range(col+1, len(current_state[row])):
    #             if current_state[col][row] != None \
    #                 and current_state[mid][row] != None \
    #                 and ((current_state[col][row] != goal_state[col][row]) or (current_state[mid][row] != goal_state[mid][row])) \
    #                 and current_state[col][row] in goal_cols[row] \
    #                 and current_state[mid][row] in goal_cols[row]:
    #                 conflicts += 2
    return conflicts


@cpu
def linear_conflicts(current_state, goal_state, size=3):
    candidate = sum(current_state, [])
    solved = sum(goal_state, [])
    def count_conflicts(candidate_row, solved_row, size, ans=0):
        counts = [0 for x in range(size)]
        for i, tile_1 in enumerate(candidate_row):
            if tile_1 in solved_row and tile_1 != None:
                solved_i = solved_row.index(tile_1)
                for j, tile_2 in enumerate(candidate_row):
                    if tile_2 in solved_row and tile_2 != None and i != j:
                        solved_j = solved_row.index(tile_2)
                        if solved_i > solved_j and i < j:
                            counts[i] += 1
                        if solved_i < solved_j and i > j:
                            counts[i] += 1
        if max(counts) == 0:
            return ans * 2
        else:
            i = counts.index(max(counts))
            candidate_row[i] = -1
            ans += 1
            return count_conflicts(candidate_row, solved_row, size, ans)

    res = 0
    candidate_rows = [[] for y in range(size)]
    candidate_columns = [[] for x in range(size)]
    solved_rows = [[] for y in range(size)]
    solved_columns = [[] for x in range(size)]
    for y in range(size):
        for x in range(size):
            idx = (y * size) + x
            candidate_rows[y].append(candidate[idx])
            candidate_columns[x].append(candidate[idx])
            solved_rows[y].append(solved[idx])
            solved_columns[x].append(solved[idx])
    for i in range(size):
        res += count_conflicts(candidate_rows[i], solved_rows[i], size)
    for i in range(size):
        res += count_conflicts(candidate_columns[i], solved_columns[i], size)
    return res


@cpu
def heuristic(current_state, goal_state):
    # return uniform_heuristic(current_state, goal_state)
    # return manhattan_heuristic(current_state, goal_state)
    # return misplaced_heuristic(current_state, goal_state)
    # return linear_conflict_heuristic(current_state, goal_state) + manhattan_heuristic(current_state, goal_state)
    return linear_conflicts(current_state, goal_state) + manhattan_heuristic(current_state, goal_state)


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
    # cpu.auto_report()

    puzzle0 = [[3, 1, 2], [7, None, 5], [4, 6, 8]]
    puzzle1 = [[7, 2, 4], [5, None, 6], [8, 3, 1]]
    puzzle2 = [[6, 7, 3], [1, 5, 2], [4, None, 8]]
    puzzle3 = [[None, 8, 6], [4, 1, 3], [7, 2, 5]]
    puzzle4 = [[7, 3, 4], [2, 5, 1], [6, 8, None]]
    puzzle5 = [[1, 3, 8], [4, 7, 5], [6, None, 2]]
    puzzle6 = [[8, 7, 6], [5, 4, 3], [2, 1, None]]
    # puzzle7 = [[2, 7, None], [5, 4, 3], [8, 1, 6]]

    puzzle8 = [[13, 10, 11, 6], [5, 7, 4, 8], [1, 12, 14, 9], [3, 15, 2, None]]
    puzzle9 = [[13, 10, 11, 6], [5, 7, 4, 8], [2, 12, 14, 9], [3, 15, 1, None]]

    goalState8 = [[None, 1, 2], [3, 4, 5], [6, 7, 8]]
    goalState15 = [[None, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]

    initial_state = State(puzzle8)
    goalState = goalState15

    RIGHT = "RIGHT"
    LEFT = "LEFT"
    UP = "UP"
    DOWN = "DOWN"
    FAILURE = "FAILURE"
    CUTOFF = "CUTOFF"

    # finalState, max_queue_size = iterate_deepening_search(initial_state, goalState)
    # finalState, max_queue_size = bfs_search(initial_state, goalState)
    finalState, max_queue_size = a_star_search(initial_state, goalState, heuristic)

    print("Initial state: ", initial_state.state)
    show_solution(finalState)
    print("Max Queue Size: ", max_queue_size)

    # print("Initial state: ", puzzle0)
    # print("Above Puzzle is solvable:", is_solvable(puzzle0))
    # print()

    # print("Initial state: ", puzzle1)
    # print("Above Puzzle is solvable:", is_solvable(puzzle1))
    # print()

    # print("Initial state: ", puzzle2)
    # print("Above Puzzle is solvable:", is_solvable(puzzle2))
    # print()

    # print("Initial state: ", puzzle3)
    # print("Above Puzzle is solvable:", is_solvable(puzzle3))
    # print()

    # print("Initial state: ", puzzle4)
    # print("Above Puzzle is solvable:", is_solvable(puzzle4))
    # print()

    # print("Initial state: ", puzzle5)
    # print("Above Puzzle is solvable:", is_solvable(puzzle5))
    # print()

    # print("Initial state: ", puzzle6)
    # print("Above Puzzle is solvable:", is_solvable(puzzle6))
    # print()

    # print("Initial state: ", puzzle8)
    # print("Above Puzzle is solvable:", is_solvable(puzzle8))
    # print()

    # print("Initial state: ", puzzle9)
    # print("Above Puzzle is solvable:", is_solvable(puzzle9))
    # print()

    # print("Initial state: ", goalState8)
    # print("Above Puzzle is solvable:", is_solvable(goalState8))
    # print()

    # print("Initial state: ", goalState15)
    # print("Above Puzzle is solvable:", is_solvable(goalState15))
    # print()
