# maze_gbfs_astar.py
import heapq
import random
import time

class Node:
    def __init__(self, position, parent=None, g=0, h=0):
        self.position = position
        self.parent = parent
        self.g = g  # Cost from start to current
        self.h = h  # Heuristic
        self.f = g + h

    def __lt__(self, other):
        return self.f < other.f

def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

def generate_grid(size, num_obstacles):
    grid = [['.' for _ in range(size)] for _ in range(size)]
    start = (0, 0)
    goal = (size - 1, size - 1)
    grid[start[0]][start[1]] = 'S'
    grid[goal[0]][goal[1]] = 'G'

    count = 0
    while count < num_obstacles:
        x, y = random.randint(0, size - 1), random.randint(0, size - 1)
        if (x, y) != start and (x, y) != goal and grid[x][y] == '.':
            grid[x][y] = '#'
            count += 1
    return grid, start, goal

def get_neighbors(pos, grid):
    neighbors = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    for dx, dy in directions:
        nx, ny = pos[0] + dx, pos[1] + dy
        if 0 <= nx < len(grid) and 0 <= ny < len(grid[0]) and grid[nx][ny] != '#':
            neighbors.append((nx, ny))
    return neighbors

def reconstruct_path(node):
    path = []
    while node:
        path.append(node.position)
        node = node.parent
    return path[::-1]

def gbfs(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, (manhattan(start, goal), Node(start)))
    visited = set()
    nodes_explored = 0

    while open_set:
        _, current = heapq.heappop(open_set)
        nodes_explored += 1
        if current.position == goal:
            return reconstruct_path(current), nodes_explored

        visited.add(current.position)
        for neighbor in get_neighbors(current.position, grid):
            if neighbor not in visited:
                h = manhattan(neighbor, goal)
                heapq.heappush(open_set, (h, Node(neighbor, current, 0, h)))

    return None, nodes_explored

def astar(grid, start, goal):
    open_set = []
    heapq.heappush(open_set, Node(start, None, 0, manhattan(start, goal)))
    visited = set()
    nodes_explored = 0

    while open_set:
        current = heapq.heappop(open_set)
        nodes_explored += 1

        if current.position == goal:
            return reconstruct_path(current), nodes_explored

        visited.add(current.position)
        for neighbor in get_neighbors(current.position, grid):
            if neighbor not in visited:
                g = current.g + 1
                h = manhattan(neighbor, goal)
                heapq.heappush(open_set, Node(neighbor, current, g, h))

    return None, nodes_explored

def run_experiments():
    sizes = [71, 224, 707, 2236, 7071]  # sqrt of number of nodes: 5000 - 50,000,000
    obstacles = [10, 100, 1000, 10000, 100000]

    print("Experiment\tTime GBFS\tTime A*\tLength GBFS\tLength A*\tNodes GBFS\tNodes A*")
    for i in range(5):
        grid, start, goal = generate_grid(sizes[i], obstacles[i])

        start_time = time.time()
        path_gbfs, nodes_gbfs = gbfs(grid, start, goal)
        time_gbfs = (time.time() - start_time) * 1000

        start_time = time.time()
        path_astar, nodes_astar = astar(grid, start, goal)
        time_astar = (time.time() - start_time) * 1000

        print(f"#{i+1}\t{int(time_gbfs)} ms\t{int(time_astar)} ms\t{len(path_gbfs) if path_gbfs else 0}\t\t{len(path_astar) if path_astar else 0}\t\t{nodes_gbfs}\t\t{nodes_astar}")

if __name__ == "__main__":
    run_experiments()
