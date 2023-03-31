import random, math, ast
from functools import reduce
from copy import copy
from builtins import isinstance
from resource import setrlimit, RLIMIT_AS, RLIMIT_DATA
from heapq import *
from time import *

def list_to_matrix(list, rows, columns):    
		"""Tranforma o lista in matrice"""
		result = []               
		start = 0
		end = columns
		res = list

		# cateodata listele vin drept list<str> in loc de list<int>
		# daca se intampla asta, le convertim la int
		if isinstance(list[0], str) and list[0] != ' ':
			res = ast.literal_eval(list)
    
		for i in range(rows): 
			result.append(res[start:end])
			start += columns
			end += columns

		return result


def get_available_moves(matrix, n):
	"""Intoarce mutarile valabile pentru o matrice"""
	available_moves = []

	for i, x in enumerate(matrix):
		if ' ' in x:
			empty_space =  i, x.index(' ')

	
	if empty_space[0] - 1 >= 0: available_moves.append(0) # sus
	if empty_space[0] + 1 < n: available_moves.append(1) # jos
	if empty_space[1] - 1 >= 0: available_moves.append(2) # stanga
	if empty_space[1] + 1 < n: available_moves.append(3) # dreapta
	
	return available_moves

class NPuzzle:
	"""
	Reprezentarea unei stări a problemei și a istoriei mutărilor care au adus starea aici.
	
	Conține funcționalitate pentru
	- afișare
	- citirea unei stări dintr-o intrare pe o linie de text
	- obținerea sau ștergerea istoriei de mutări
	- obținerea variantei rezolvate a acestei probleme
	- verificarea dacă o listă de mutări fac ca această stare să devină rezolvată.
	"""

	NMOVES = 4
	UP, DOWN, LEFT, RIGHT = range(NMOVES)
	ACTIONS = [UP, DOWN, LEFT, RIGHT]
	names = "UP, DOWN, LEFT, RIGHT".split(", ")
	BLANK = ' '
	delta = dict(zip(ACTIONS, [(-1, 0), (1, 0), (0, -1), (0, 1)]))
	
	PAD = 2
	
	def __init__(self, puzzle : list[int | str], movesList : list[int] = []):
		"""
		Creează o stare nouă pe baza unei liste liniare de piese, care se copiază.
		
		Opțional, se poate copia și lista de mutări dată.
		"""
		self.N = len(puzzle)
		self.side = int(math.sqrt(self.N))
		self.r = copy(puzzle)
		self.moves = copy(movesList)
		self.solved_puzzle = list(range(1, self.N)) + [' ']
		self.solved_matrix = list_to_matrix(self.solved_puzzle, self.side, self.side)

	def display(self, show = True) -> str:
		l = "-" * ((NPuzzle.PAD + 1) * self.side + 1)
		aslist = self.r
		
		slices = [aslist[ slice * self.side : (slice+1) * self.side ]  for slice in range(self.side)]
		s = ' |\n| '.join([' '.join([str(e).rjust(NPuzzle.PAD, ' ') for e in line]) for line in slices]) 
	
		s = ' ' + l + '\n| ' + s + ' |\n ' + l
		if show: print(s)
		return s
	def display_moves(self):
		print([self.names[a] if a is not None else None for a in self.moves])
		
	def print_line(self):
		return str(self.r)
	
	@staticmethod
	def read_from_line(line : str):
		list = line.strip('\n][').split(', ')
		numeric = [NPuzzle.BLANK if e == "' '" else int(e) for e in list]
		return NPuzzle(numeric)
	
	def clear_moves(self):
		"""Șterge istoria mutărilor pentru această stare."""
		self.moves.clear()
	
	def apply_move_inplace(self, move : int):
		"""Aplică o mutare, modificând această stare."""
		blankpos = self.r.index(NPuzzle.BLANK)
		y, x = blankpos // self.side, blankpos % self.side
		ny, nx = y + NPuzzle.delta[move][0], x + NPuzzle.delta[move][1]
		if ny < 0 or ny >= self.side or nx < 0 or nx >= self.side: return None
		newpos = ny * self.side + nx
		piece = self.r[newpos]
		self.r[blankpos] = piece
		self.r[newpos] = NPuzzle.BLANK
		self.moves.append(move)
		return self
	
	def apply_move(self, move : int):
		"""Construiește o nouă stare, rezultată în urma aplicării mutării date."""
		return self.clone().apply_move_inplace(move)

	def solved(self):
		"""Întoarce varianta rezolvată a unei probleme de aceeași dimensiune."""
		return NPuzzle(list(range(self.N))[1:] + [NPuzzle.BLANK])

	def verify_solved(self, moves : list[int]) -> bool:
		""""Verifică dacă aplicarea mutărilor date pe starea curentă duce la soluție"""
		return reduce(lambda s, m: s.apply_move_inplace(m), moves, self.clone()) == self.solved() # type: ignore

	def out_of_place_heuristic(self):
		"""Intoarce numarul de piese ce nu sunt in pozitia finala"""
		h = 0
		for i in range(self.N):
			if self.solved_puzzle[i] == self.r[i]:
				h += 1
			
		return len(self.solved_puzzle) - h	

	def manhattan_distance(self):
		"""Intoarce numarul corespunzator sumei distantelor Manhattan pentru fiecare piesa dintre pozitia actuala si cea finala"""
		h = 0
		position = []
		matrix = list_to_matrix(self.r, self.side, self.side)
	
		for i in range(self.side):
			for j in range(self.side):
				position = [(index, row.index(matrix[i][j])) for index, row in enumerate(self.solved_matrix) if matrix[i][j] in row]
				
				x = position[0][0]
				y = position[0][1]

				h += abs(i - x) + abs(j - y)
				
		return h

	def astar(self, initial, goal):
		"""Intoarce un tuplu format din (adancimea arborelui de joc, numarul de stari stocate)"""
		frontier = []
		startNode = (0, initial)

		# punem starea initiala in frontiera
		# starea initiala = (cost, vectorul jocului)
		heappush(frontier, startNode)

		# o adaugam in discover
		discovered = {str(initial): (0, 0)}

		# cat timp mai avem noduri de explorat
		while frontier:
			
			# scoatem primul nod
			crt_node = heappop(frontier)[1]

			# ii luam costul
			crt_cost = discovered[str(crt_node)][1]

			# daca numarul de stari stocate trece peste limita impusa returnam -1
			if len(discovered) > 100000:
				return -1

			# daca ajungem la starea finala jocul a fost rezolvat
			# intoarcem tuplul (distanta, stari_stocate)
			if str(crt_node) == str(goal):
				return (discovered[str(crt_node)][0], len(discovered))

			# transformam lista intr-o matrice
			crt_node_matrix = list_to_matrix(crt_node, self.side, self.side)
			
			# obtinem mutarile posibile pentru nodul curent
			moves = get_available_moves(crt_node_matrix, self.side)
			
			# cream un puzzle auxiliar cu nodul nostru
			aux_puzzle = NPuzzle.read_from_line(str(crt_node))
			
			# cream o lista de vecini
			list_neigh = []
			
			# jucam fiecare mutare si adaugam noile stari in lista de vecini
			for move in moves:
				list_neigh.append(aux_puzzle.apply_move(move))
			
			# pentru fiecare vecin verificam daca il putem adauga in frontiera
			for neigh in list_neigh:
				
				# incrementam costul de la nodul initial la cel curent
				cost_neigh = crt_cost + 1

				# obtinem lista pentru jocul curent din lista de vecini
				curr_list = neigh.r

				# daca nu e in discovered sau are un cost mai bun
				if str(curr_list) not in discovered or discovered[str(curr_list)][1] > cost_neigh:

					# actualizam discovered pentru acest nod cu costul acesta
					discovered[str(curr_list)] = (discovered[str(crt_node)][0] + 1, cost_neigh)

					# calculam nou cost adaugan euristica aleasa
					new_cost = cost_neigh + neigh.manhattan_distance()

					# adaugam nodul in frontiera
					heappush(frontier, (new_cost, str(curr_list)))
		return -1

	def beamsearch(self, initial, goal, B):
		"""Intoarce un tuplu format din (adancimea arborelui de joc, numarul de stari stocate)"""
		initial_node = initial
		# beam functioneaza similar frontierei de la astar
		beam = []

		# adaugam in beam nodul initial
		heappush(beam, str(initial_node))

		# adaugam in discovered nodul initial si un tuplu (adancime arbore, cost pana la stare)
		discovered = {str(initial_node): (0, 0)}

		# cat timp mai avem elemente in beam sau nu am atins limita de stari
		while beam and len(discovered) < 1000000:
			succ = {}

			# setam latimea beamului
			beam_width = B

			while beam:
				# scoatem din beam primul element
				crt_node = heappop(beam)
				
				# cream un puzzle auxiliar cu nodul nostru
				aux_puzzle = NPuzzle.read_from_line(str(crt_node))
				
				# transformam lista intr-o matrice
				crt_node_matrix = list_to_matrix(crt_node, self.side, self.side)
				
				# obtinem mutarile posibile pentru nodul curent
				moves = get_available_moves(crt_node_matrix, self.side)
				
				# cream o lista de vecini
				list_neigh = []
				
				# jucam fiecare mutare si adaugam noile stari in lista de vecini
				for move in moves:
					list_neigh.append(aux_puzzle.apply_move(move))

				# verificam fiecare vecin din lista de vecini
				for neigh in list_neigh:
					# daca am ajuns la starea finala intoarcem tuplul
					if neigh.r == goal:
						return (discovered[crt_node][0] + 1, len(discovered) + 1)
					
					# daca vecinul nu este in discovered il adaugam in lista de succesori 
					# cu costul asociat euristicii folosite
					if str(neigh.r) not in discovered:
						succ[str(neigh.r)] = neigh.manhattan_distance()
			
			# sortam succesorii dupa costul asociat fiecarui nod
			succ = sorted(succ.items(), key=lambda h: h[1])

			# selectam in discovered primele B noduri si le punem in beam
			for node in succ:
				if beam_width > 0:
					discovered[node[0]] = (discovered[crt_node][0] + 1, node[1])
					heappush(beam, node[0])
					beam_width = beam_width -1
		
		return -1
			
			


	

	def clone(self):
		return NPuzzle(self.r, self.moves)
	def __str__(self) -> str:
		return str(self.N-1) + "-puzzle:" + str(self.r)
	def __repr__(self) -> str: return str(self)
	def __eq__(self, other):
		return self.r == other.r
	def __lt__(self, other):
		return True
	def __hash__(self):
		return hash(tuple(self.r))
	


MLIMIT = 3 * 10 ** 9 # 2 GB RAM limit
setrlimit(RLIMIT_DATA, (MLIMIT, MLIMIT))

f = open("files/problems4.txt", "r")
input = f.readlines()
f.close()
problems = [NPuzzle.read_from_line(line) for line in input]
# problems[0].display()
# print(problems[0].r)
# print(problems[0].N)
# print(problems[0].matrix)
# print(problems[0].solved_puzzle)
# print(problems[0].solved_matrix)


# generare
def genOne(side, difficulty):
	state = NPuzzle(list(range(side * side))[1:] + [NPuzzle.BLANK])
	for i in range(side ** difficulty + random.choice(range(side ** (difficulty//2)))):
		s = state.apply_move(random.choice(NPuzzle.ACTIONS))
		if s is not None: state = s
	state.clear_moves()
	return state


# print("Generare:")
random.seed(4242)
# p = genOne(3, 3)
# print(p.astar(p.r, p.solved_puzzle))

aux = problems[0].apply_move(0)
B_wid = [1, 10, 50, 100, 500, 1000]
for b in B_wid:
	print("B = ", b)
	for problem in problems:
		start = time()
		# ceva = problem.astar(problem.r, problem.solved_puzzle)
		ceva = problem.beamsearch(problem.r, problem.solved_puzzle, b)
		end = time()
		if isinstance(ceva, int):
			print(ceva)
		else:
			x, y = ceva
			print(x)
		
		# print("Time elapsed: ")
		# print(end - start)
		# print("======")


# problemele easy au fost generate cu dificultatile 4, 3, respectiv 2 (pentru marimile 4, 5, 6)
# celelalte probleme au toate dificultate 6

