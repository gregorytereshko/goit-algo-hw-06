import networkx as nx
import matplotlib.pyplot as plt
from collections import deque

# Створення графа, що моделює транспортну мережу маленького міста
def create_graph():
  G = nx.Graph()

  # Додавання вершин (зупинок)
  stops = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
  G.add_nodes_from(stops)

  # Додавання ребер (маршрутів між зупинками)

  G.add_edges_from([
    ('A', 'B', {'weight': 2}),
    ('A', 'C', {'weight': 5}),
    ('B', 'D', {'weight': 3}),
    ('C', 'D', {'weight': 2}),
    ('D', 'E', {'weight': 1}),
    ('D', 'G', {'weight': 6}),
    ('E', 'F', {'weight': 4}),
    ('F', 'G', {'weight': 1}),
    ('G', 'H', {'weight': 5}),
    ('H', 'I', {'weight': 3}),
    ('I', 'J', {'weight': 2}),
    ('J', 'A', {'weight': 7})
  ])

  return G

# Візуалізація графа
def draw_graph(G):
  plt.figure(figsize=(8, 8))
  pos = nx.spring_layout(G)  # Задаємо розташування вершин
  nx.draw(G, pos, with_labels=True, node_size=700, node_color='lightblue', font_size=16, font_weight='bold')
  edge_labels = nx.get_edge_attributes(G, 'weight')
  nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels, font_size=14)
  plt.title('Транспортна мережа маленького міста')
  plt.show()

def dfs(graph, start, goal):
  stack = [(start, [start])]
  while stack:
    (vertex, path) = stack.pop()
    for next in set(graph[vertex]) - set(path):
      if next == goal:
        return path + [next]
      else:
        stack.append((next, path + [next]))

def bfs(graph, start, goal):
  queue = deque([(start, [start])])
  while queue:
    (vertex, path) = queue.popleft()
    for next in set(graph[vertex]) - set(path):
      if next == goal:
        return path + [next]
      else:
        queue.append((next, path + [next]))

def dijkstra(graph, start):
  # Ініціалізація відстаней та множини невідвіданих вершин
  distances = {vertex: float('infinity') for vertex in graph.nodes}
  distances[start] = 0
  unvisited = set(graph.nodes)

  while unvisited:
    # Знаходження вершини з найменшою відстанню серед невідвіданих
    current_vertex = min(unvisited, key=lambda vertex: distances[vertex])

    # Якщо поточна відстань є нескінченністю, то ми завершили роботу
    if distances[current_vertex] == float('infinity'):
      break

    for neighbor in graph.neighbors(current_vertex):
      weight = graph[current_vertex][neighbor].get('weight', 1)  # Використовуємо вагу 1, якщо вона не вказана
      distance = distances[current_vertex] + weight

      # Якщо нова відстань коротша, то оновлюємо найкоротший шлях
      if distance < distances[neighbor]:
        distances[neighbor] = distance

    # Видаляємо поточну вершину з множини невідвіданих
    unvisited.remove(current_vertex)

  return distances

G = create_graph()

# Аналіз основних характеристик
num_nodes = G.number_of_nodes()
num_edges = G.number_of_edges()
degrees = [degree for node, degree in G.degree()]

print(f'Кількість вершин: {num_nodes}')
print(f'Кількість ребер: {num_edges}')
print(f'Ступені вершин: {degrees}')

start_node = 'A'
goal_node = 'H'
dfs_path = dfs(G, start_node, goal_node)
bfs_path = bfs(G, start_node, goal_node)

print('Початкова нода:', start_node)
print('Кінцева нода:', goal_node)
print(f'Шлях з використанням DFS: {dfs_path}')
print(f'Шлях з використанням BFS: {bfs_path}')

shortest_paths = dijkstra(G, 'A')
print("Найкоротші шляхи від вершини A до всіх інших вершин:")
for vertex, distance in shortest_paths.items():
    print(f"Відстань до вершини {vertex}: {distance}")

draw_graph(G)