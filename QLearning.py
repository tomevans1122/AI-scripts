import numpy as np 


#Setting the gamma and alpha parameters for Q learning
gamma = 0.75
alpha = 0.9


#BUILDING THE ENVIRONMENT
# Maze is made up of letters from A to L, thus 11 possible states
location_to_state = {'A': 0,
					 'B': 1,
					 'C': 2,
					 'D': 3,
					 'E': 4,
					 'F': 5,
					 'G': 6,
					 'H': 7,
					 'I': 8,
					 'J': 9,
					 'K': 10,
					 'L': 11}

# Possible actions made by Q learning is also list of integers up to 11
actions = [0,1,2,3,4,5,6,7,8,9,10,11]

# Make a matrix of rewards- columns are actions, rows are states
R = np.array([[0,1,0,0,0,0,0,0,0,0,0,0],
              [1,0,1,0,0,1,0,0,0,0,0,0],
              [0,1,0,0,0,0,1,0,0,0,0,0],
              [0,0,0,0,0,0,0,1,0,0,0,0],
              [0,0,0,0,0,0,0,0,1,0,0,0],
              [0,1,0,0,0,0,0,0,0,1,0,0],
              [0,0,1,0,0,0,1000,1,0,0,0,0],
              [0,0,0,1,0,0,1,0,0,0,0,1],
              [0,0,0,0,1,0,0,0,0,1,0,0],
              [0,0,0,0,0,1,0,0,1,0,1,0],
              [0,0,0,0,0,0,0,0,0,1,0,1],
              [0,0,0,0,0,0,0,1,0,0,1,0]])



# BUILDING THE AI

#Initializing Q-values
Q = np.array(np.zeros([12,12]))

#Implementing the Q-learning process
for i in range(1000):
	current_state = np.random.randint(0,12)
	playable_actions = []

	for j in range(12):
		if R[current_state, j] > 0:
			playable_actions.append(j)
	
	next_state = np.random.choice(playable_actions)
	#Temporal difference
	TD = R[current_state, next_state] + gamma*Q[next_state, np.argmax(Q[next_state,])] - Q[current_state, next_state] 

	# Q value
	Q[current_state, next_state] = Q[current_state, next_state] + alpha * TD 


	print("Q values: ")
	print(Q.astype(int))
