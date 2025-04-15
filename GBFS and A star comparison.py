import heapq
import time

grid_str = [
    "S..#......",
    ".#.#.####.",
    ".#......#.",
    ".#####..#.",
    ".....#..#G",
    "####.#..##",
    "...#.#....",
    ".#.#.####.",
    ".#........",
    "....#####."
]

# Parse grid
grid = [list(row) for row in grid_str]
rows, cols = len(grid), len(grid[0])

# Find start and goal
start = goal = None
for r in range(rows):
    for c in range(cols):
        if grid[r][c] == 'S':
            start = (r, c)
        elif grid[r][c] == 'G':
            goal = (r, c)

# Heuristic: Manhattan
def manhattan(p1, p2):
    return abs(p1[0] - p2[0]) + abs(p1[1] - p2[1])

# Neighbors
def neighbors(pos):
    r, c = pos
    for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
        nr, nc = r + dr, c + dc
        if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != '#':
            yield (nr, nc)

# Path reconstruction
def reconstruct(came_from, current):
    path = []
    while current in came_from:
        path.append(current)
        current = came_from[current]
    return path[::-1]

# GBFS implementation
def gbfs(start, goal):
    frontier = [(manhattan(start, goal), start)]
    came_from = {}
    visited = set()
    nodes_explored = 0

    while frontier:
        _, current = heapq.heappop(frontier)
        nodes_explored += 1

        if current == goal:
            return reconstruct(came_from, current), nodes_explored

        visited.add(current)
        for neighbor in neighbors(current):
            if neighbor not in visited:
                heapq.heappush(frontier, (manhattan(neighbor, goal), neighbor))
                if neighbor not in came_from:
                    came_from[neighbor] = current

    return [], nodes_explored

# A* implementation
def astar(start, goal):
    frontier = [(manhattan(start, goal), 0, start)]
    came_from = {}
    cost_so_far = {start: 0}
    nodes_explored = 0

    while frontier:
        _, g, current = heapq.heappop(frontier)
        nodes_explored += 1

        if current == goal:
            return reconstruct(came_from, current), nodes_explored

        for neighbor in neighbors(current):
            new_cost = g + 1
            if neighbor not in cost_so_far or new_cost < cost_so_far[neighbor]:
                cost_so_far[neighbor] = new_cost
                priority = new_cost + manhattan(neighbor, goal)
                heapq.heappush(frontier, (priority, new_cost, neighbor))
                came_from[neighbor] = current

    return [], nodes_explored

# Run and compare
start_time = time.time()
path_gbfs, nodes_gbfs = gbfs(start, goal)
time_gbfs = (time.time() - start_time) * 1000

start_time = time.time()
path_astar, nodes_astar = astar(start, goal)
time_astar = (time.time() - start_time) * 1000

# Visualization
def print_path(path):
    path_set = set(path)
    for r in range(rows):
        line = ''
        for c in range(cols):
            if (r, c) == start:
                line += 'S'
            elif (r, c) == goal:
                line += 'G'
            elif (r, c) in path_set:
                line += '*'
            else:
                line += grid[r][c]
        print(line)

print("GBFS Path:")
print_path(path_gbfs)
print(f"Path length: {len(path_gbfs)} | Nodes explored: {nodes_gbfs} | Time: {time_gbfs:.2f}ms\n")

print("A* Path:")
print_path(path_astar)
print(f"Path length: {len(path_astar)} | Nodes explored: {nodes_astar} | Time: {time_astar:.2f}ms")
