import random
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.animation import FuncAnimation
import utils
import math


class Node:
    """ Clasa folosita pentru reprezentarea unui nod din arborele algoritmului RRT """

    def __init__(self, pos, parent=None):
        self.pos = pos
        self.parent = parent

        if parent is not None:
            self.cost = utils.euclidean_distance(self.parent.pos, pos) + self.parent.cost
        else:
            self.cost = 0


class RRT:
    """ Clasa folosita pentru reprezentarea algoritmului RRT / RRT* """

    def __init__(self, agent_index, RRT_Star, start, goal, width, height, obstacles, stop_max_nodes, num_nodes,
                  step_size, neigh_radius, eps_distance):
        
        self.agent_index = agent_index
        self.RRT_Star = RRT_Star
        self.start = start
        self.goal = goal
        self.width = width
        self.height = height
        self.obstacles = obstacles
        self.stop_max_nodes = stop_max_nodes
        self.num_nodes = num_nodes
        self.step_size = step_size
        self.neigh_radius = neigh_radius
        self.eps_distance = eps_distance

        self.tree = [Node(start)]
        self.path = []
        self.found_path = False

    def steer(self, from_node, to_point):
        """ Functie care primeste un nod din arbore si o pozitie si genereaza un nod nou in directia from_node -> to_point
            la distanta self.step_size fata de from_node """
        
        ### TODO 1

        ### TODO 1.1
        # folosind functia get_direction din utils.py se calculeaza directia from_node -> to_point
        if from_node.pos[0]!=to_point[0] or from_node.pos[1] != to_point[1]:
            direction = utils.get_direction(from_node.pos, to_point)
        else:
            direction = 0
        theta = math.atan2(direction[1], direction[0])
        ### TODO 1.2
        # se calculeaza pozitia care este la distanta self.step_size fata de from_nod si pe directia noul calculata
        new_pos = (from_node.pos[0] + self.step_size*math.cos(theta), from_node.pos[1] + self.step_size*math.sin(theta))
        return Node(new_pos, from_node)

    def find_nearby_nodes(self, new_node):
        """ Functie care primeste un nod si genereaza lista de noduri care se afla la cel mult self.neigh_radius fata de nod """

        ### TODO 2

        # folosind functia euclidean_distance din utils.py se genereaza lista de noduri vecine
        list_elements = []
        for node in self.tree:
            dist = utils.euclidean_distance(node.pos, new_node.pos)
            if dist <= self.neigh_radius:
                list_elements.append(node)

        return list_elements

    def is_collision_free(self, parent_node, new_node):
        """ Functie care primeste doua noduri si verifica daca drumul dintre cele doua genereaza coliziuni """

        # se verifica limitele hartii
        if new_node.pos[0] < 0 or new_node.pos[0] >= self.width or new_node.pos[1] < 0 or new_node.pos[1] >= self.height:
            return False

        start_pos = parent_node.pos
        end_pos = new_node.pos

        # se verifica fiecare obstacol din mediu
        for obst in self.obstacles:
            rect = (obst[0][0], obst[0][1], obst[1][0], obst[1][1])

            if utils.line_intersects_rectangle(start_pos, end_pos, rect):
                return False

        return True

    def rewire(self, new_node, nearby_nodes):
        """ Functie folosita de RRT* care primeste un nod (new_node) si lista nodurilor vecine si incearca sa
            reconfigureze aceste noduri. Reconfigurarea unui nod inseamna schimbarea parintelui curent al acelui
            nod cu new_node atunci cand utilizarea new_node ca parinte ofera un cost mai mic """

        ### TODO 3

        for node in nearby_nodes:
            if new_node == node or new_node.parent == node:
                continue

            ### TODO 3.1
            # se calculeaza costul potential daca ar fi folosit new_node ca parinte a nodului actual
            potential_cost = utils.euclidean_distance(new_node.pos, node.pos) + new_node.cost

            ### TODO 3.2
            # daca costul potential este mai mic decat costul actual al nodului se reconfigureaza nodul curent prin
            # schimbarea parintelui in new_node
            if potential_cost < node.cost and self.is_collision_free(new_node, node):
                node.cost = potential_cost
                node.parent = new_node


    def expand_tree(self):
        """ Functia principala folosita de algoritmul RRT in extinderea arborelui curent. Aceasta functie alege aleator
            o pozitie din harta si incearca conectarea acesteia la arbore"""

        ### TODO 4

        ### TODO 4.1
        # alegerea unei pozitii aleatoare
        rand_point = (random.randint(0, 300), random.randint(0, 300))

        ### TODO 4.2
        # gasirea nodului din arborele curent care este cel mai apropiat de pozitia aleasa
        cost = math.inf
        best_node = None
        for node in self.tree:
            dist = utils.euclidean_distance(node.pos, rand_point)
            if dist < cost:
                cost = dist
                best_node = node

        nearest_node = best_node

        # generarea unui nod nou folosind functia steer
        new_node = self.steer(nearest_node, rand_point)

        # verificare daca noul nod genereaza vreo coliziune cu mediul
        if self.is_collision_free(nearest_node, new_node):

            # pentru cazul folosirii algoritmului RRT*
            if self.RRT_Star:
                # se genereaza toti vecinii nodului curent
                nearby_nodes = self.find_nearby_nodes(new_node)

                if nearby_nodes:
                    ### TODO 4.3
                    # se determina nodul (dintre nodurile vecine) care daca ar fi parintele nodului curent ar da un cost minim
                    minDist = math.inf
                    node_chosen = None
                    for node in nearby_nodes:
                        dist = utils.euclidean_distance(new_node.pos, node.pos) + node.cost
                        if dist < minDist:
                            minDist = dist
                            node_chosen = node
                    
                    min_cost_node = node_chosen
                    
                    ### TODO 4.4
                    # se calculeaza costul potential
                    potential_cost = utils.euclidean_distance(min_cost_node.pos, new_node.pos) + min_cost_node.cost

                    ### TODO 4.5
                    # daca legarea nodului nou la nodul min_cost_node da un cost mai mic decat costul actual al noului nod
                    # se seteaza min_cost_node ca parintele noului nod
                    if potential_cost < new_node.cost and self.is_collision_free(new_node, min_cost_node):
                        new_node.cost = potential_cost
                        new_node.parent = min_cost_node

            ### TODO 4.6
            # se adauga noul nod in arbore (in lista de noduri)
            # ...
            self.tree.append(new_node)

            # pentru cazul folosirii algoritmului RRT*
            if self.RRT_Star:
                # se face rewire la nodurile din apropierea noului nod
                self.rewire(new_node, nearby_nodes)

            # daca noul nod este la cel mult self.eps_distance de goal se opreste cautarea
            if utils.euclidean_distance(new_node.pos, self.goal) < self.eps_distance:
                self.found_path = True
                self.path = self.retrace_path(new_node)

    def retrace_path(self, node):
        """ Functie care reconstruieste drumul de la start la goal """

        ### TODO 5

        path = []

        # ...
        path.append(node)
        while node!=None and node!=self.start:
            path.append(node.parent)
            node = node.parent
        path.reverse()
        return path

    def build_rrt(self):
        """ Functie care construieste arborele RRT prin apelarea iterativa a functie de extindere a arborului curent """

        index = 0
        while True:
            if self.stop_max_nodes and index >= self.num_nodes:
                break

            if not self.found_path:
                self.expand_tree()
                index += 1
            else:
                break
        
        if self.found_path:
            print(f"Agent {self.agent_index} found a valid path in {index} steps.")
        else:
            print(f"Agent {self.agent_index} did NOT find a valid path in {index} steps.")


class MultiAgentRRT:
    def __init__(self, config_dict):
        self.tree_lines = None
        self.agents = []
        self.config_dict = config_dict

        for i in range(config_dict["num_agents"]):
            start = self.config_dict["start_pos"][i]
            goal = self.config_dict["end_pos"][i]
            rrt = RRT(i, self.config_dict["RRT_Star"], start, goal, self.config_dict["size"][0], self.config_dict["size"][1],
                        self.config_dict["obstacles"], self.config_dict["stop_max_nodes"], self.config_dict["num_nodes"],
                        self.config_dict["step_size"], self.config_dict["neigh_radius"], self.config_dict["eps_dist"])
            self.agents.append(rrt)

        for i, agent in enumerate(self.agents):
            agent.build_rrt()

    def visualize_trees(self):
        plt.rcParams['figure.figsize'] = [9, 9]

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.config_dict["size"][0])
        ax.set_ylim(0, self.config_dict["size"][1])

        start_circles = []
        goal_circles = []
        agent_circles = []

        self.tree_lines = [[] for _ in range(self.config_dict["num_agents"])]

        for obst in self.config_dict["obstacles"]:
            min_x, min_y = obst[0]
            max_x, max_y = obst[1]

            width = max_x - min_x
            height = max_y - min_y

            rect = Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='y', facecolor='y', zorder=0)
            ax.add_artist(rect)
            

        for i, agent in enumerate(self.agents):
            start_circle = plt.Circle(agent.start, 2, color='black', zorder=10)
            goal_circle = plt.Circle(agent.goal, 2, color=self.config_dict["agent_colors"][i], zorder=10)
            agent_circle = plt.Circle(agent.start, 2, color=self.config_dict["agent_colors"][i], zorder=10)

            start_circles.append(start_circle)
            goal_circles.append(goal_circle)
            agent_circles.append(agent_circle)

            agent.tree = sorted(agent.tree, key=lambda x: x.cost)

            for node in agent.tree:
                if node.parent:
                    line = plt.Line2D([node.pos[0], node.parent.pos[0]], [node.pos[1], node.parent.pos[1]],
                                       color=self.config_dict["agent_colors"][i], alpha=0.2, zorder=1)
                    ax.add_artist(line)
                    self.tree_lines[i].append(line)
                    line.set_visible(False)

            ax.add_artist(start_circle)
            ax.add_artist(goal_circle)
            ax.add_artist(agent_circle)

        def update(frame):
            for i in range(self.config_dict["num_agents"]):
                if frame < len(self.tree_lines[i]):
                    self.tree_lines[i][frame].set_visible(True)

        anim = FuncAnimation(fig, update, frames=max(len(agent.tree) for agent in self.agents), interval=1, repeat=True)
        plt.show()

    def visualize_path(self):
        plt.rcParams['figure.figsize'] = [9, 9]

        fig, ax = plt.subplots()
        ax.set_xlim(0, self.config_dict["size"][0])
        ax.set_ylim(0, self.config_dict["size"][1])

        start_circles = []
        goal_circles = []
        agent_circles = []

        self.tree_lines = [[] for _ in range(self.config_dict["num_agents"])]

        for obst in self.config_dict["obstacles"]:
            min_x, min_y = obst[0]
            max_x, max_y = obst[1]

            width = max_x - min_x
            height = max_y - min_y

            rect = Rectangle((min_x, min_y), width, height, linewidth=1, edgecolor='y', facecolor='y', zorder=0)
            ax.add_artist(rect)

        for i, agent in enumerate(self.agents):
            start_circle = plt.Circle(agent.start, 2, color='black', zorder=10)
            goal_circle = plt.Circle(agent.goal, 2, color=self.config_dict["agent_colors"][i], zorder=10)
            agent_circle = plt.Circle(agent.start, 2, color=self.config_dict["agent_colors"][i], zorder=10)

            start_circles.append(start_circle)
            goal_circles.append(goal_circle)
            agent_circles.append(agent_circle)

            agent.tree = sorted(agent.tree, key=lambda x: x.cost)

            for node in agent.tree:
                if node.parent:
                    line = plt.Line2D([node.pos[0], node.parent.pos[0]], [node.pos[1], node.parent.pos[1]],
                                       color=self.config_dict["agent_colors"][i], alpha=0.2, zorder=1)
                    ax.add_artist(line)
                    self.tree_lines[i].append(line)

            if agent.found_path:
                for i in range(len(agent.path) - 1):
                    path_line = plt.Line2D([agent.path[i][0], agent.path[i + 1][0]], [agent.path[i][1],
                                            agent.path[i + 1][1]], color='orange', zorder=5, linewidth = 2.5)
                    ax.add_artist(path_line)

            ax.add_artist(start_circle)
            ax.add_artist(goal_circle)
            ax.add_artist(agent_circle)

        def update(frame):
            if frame == 10:
                for i in range(self.config_dict["num_agents"]):
                    for line in self.tree_lines[i]:
                        line.set_visible(False)

            for agent, agent_circle in zip(self.agents, agent_circles):
                if frame < len(agent.path):
                    agent_circle.center = agent.path[frame]

        anim = FuncAnimation(fig, update, frames=max(len(agent.tree) for agent in self.agents), interval=200, repeat=True)
        plt.show()


def apply_size_config(config_dict):
    x_factor, y_factor = config_dict['size']

    config_dict['start_pos'] = [[x * x_factor, y * y_factor] for x, y in config_dict['start_pos_norm']]
    config_dict['end_pos'] = [[x * x_factor, y * y_factor] for x, y in config_dict['end_pos_norm']]
    config_dict['obstacles'] = [[[x_min * x_factor, y_min * y_factor], [x_max * x_factor, y_max * y_factor]]
                                 for [x_min, y_min], [x_max, y_max] in config_dict['obstacles_norm']]

    return config_dict


def main():
    config_dict = {
        'num_agents' : 3,
        'agent_colors' : ['green', 'blue', 'red'],

        'start_pos_norm' : [[0.2, 0.1], [0.5, 0.1], [0.8, 0.1]],
        'end_pos_norm'  : [[0.8, 0.9], [0.5, 0.9], [0.2, 0.9]],
        'obstacles_norm' : [[[0.4, 0.4], [0.6, 0.6]], [[0.1, 0.2], [0.24, 0.4]], [[0.56, 0.7], [0.92, 0.83]]],
        'size' : [300, 300],

        'step_size' : 4,
        'neigh_radius' : 20,
        'eps_dist' : 5,

        'stop_max_nodes' : True,
        'num_nodes' : 3000,
        'RRT_Star' : True
    }

    config_dict = apply_size_config(config_dict)

    multi_rrt = MultiAgentRRT(config_dict)
    multi_rrt.visualize_trees()
    multi_rrt.visualize_path()

if __name__ == "__main__":
    main()
