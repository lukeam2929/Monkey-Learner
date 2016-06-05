# Imports.
import numpy as np
import numpy.random as npr

from SwingyMonkey import SwingyMonkey


class Learner(object):
	'''
	This agent jumps randomly.
	'''

	def __init__(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None

		self.isFirstState = True # to catch weirdness for first state of game
		self.isSecondState = False

		self.Q = {} # initialize Q table. use a dictionary for now
		self.gamma = 0.7 # temporal discount value, not too high since don't expect many duplicate states
		self.eps = 0.2 # do random action 5% of the time, for exploration?
		self.g = npr.choice([1,4]) # make a random guess at the gravity
		self.alpha = 0.7

		#self.lowTop = 0	

	def reset(self):
		self.last_state  = None
		self.last_action = None
		self.last_reward = None
		self.g = npr.choice([1,4])
		self.eps = max([0.01, self.eps-0.001])
		self.isFirstState = True
		self.isSecondState = False

	def hash_state(self, state):
		# hashes a state dictionary to a Q-table index using relevant features
		# monkey velocity, and r_x, r_y between monkey and tree

		v = state['monkey']['vel']
		HorizDist = state['tree']['dist']
		TreebotDist = state['monkey']['bot']-state['tree']['bot']
		Treetop = state['tree']['top']

		# coarse-grain position here, if necessary?

		dv = 1
		dx = 1
		dy = 1
		rv = np.round(float(v)/float(dv))
		rHor = np.round(float(HorizDist)/float(dx))
		rBot = np.round(float(TreebotDist)/float(dy))

		return [rv, rHor, rBot, Treetop]


	def action_callback(self, state):
		'''
		Implement this function to learn things and take actions.
		Return 0 if you don't want to jump and 1 if you do.
		'''

		alpha = self.alpha

		# first, just sit tight if it's the first state
		if self.isFirstState:
			self.last_state = state
			self.last_action = 0
			# move to the second state
			self.isFirstState = False
			self.isSecondState = True
			# don't jump on the first state
			return 0

		## find hash values for different states
		s_old = self.hash_state(self.last_state)
		s_new = self.hash_state(state)
		
		a_old = self.last_action

		# calculate the gravity
		if self.isSecondState:
			self.g = -state['monkey']['vel']
			self.isSecondState = False

		'''
		don't jump if the top tree is really low
		if s_old[3] < 225:
			self.lowTop +=1
			print self.lowTop
			return 0
		'''

		# get value to update from old state
		try:
			Q0 = self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old]
		except KeyError:
			Q0 = 0 # if no value there yet

		# find max Q value at new / current state, as well as next action to return
		try:
			QnewStay = self.Q[self.g][s_new[0]][s_new[1]][s_new[2]][0] # Q value for not jumping
		except KeyError:
			QnewStay = 0
		try:
			QnewJump = self.Q[self.g][s_new[0]][s_new[1]][s_new[2]][1] # Q value for jumping
		except KeyError:
			QnewJump = 0

		if QnewJump > QnewStay: # if Q value higher for jumping
			new_action = 1
			Qmax = QnewJump
		else: # otherwise, don't jump
			new_action = 0
			Qmax = QnewStay

		# epsilon greedy: with probability epsiolon, overwrite new action w random one
		if npr.rand() < self.eps: # then choose randomly 
			new_action = npr.rand() < 0.5

		# and update Q value
		# will need layered try-catch blocks to handle exceptions when the dictorary hashes don't exist yet
		# start trying to hash into actual state, then go backwards from there
		
		try:
			self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old] = (1-alpha)*Q0 + alpha*(self.last_reward + self.gamma*Qmax)
			
		except KeyError: # if we cannot hash in here?
			
			try: # try one level up
				self.Q[self.g][s_old[0]][s_old[1]][s_old[2]] = {}
				self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old] = (1-alpha)*Q0 + alpha*(self.last_reward + self.gamma*Qmax)
				
			except KeyError: # if can't hash here either?
				
				try:
					self.Q[self.g][s_old[0]][s_old[1]] = {}
					self.Q[self.g][s_old[0]][s_old[1]][s_old[2]] = {}
					self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old] = (1-alpha)*Q0 + alpha*(self.last_reward + self.gamma*Qmax)
					
				except KeyError: # etc....
					try:
						self.Q[self.g][s_old[0]] = {}
						self.Q[self.g][s_old[0]][s_old[1]] = {}
						self.Q[self.g][s_old[0]][s_old[1]][s_old[2]] = {}
						self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old] = (1-alpha)*Q0 + alpha*(self.last_reward + self.gamma*Qmax)
					except KeyError:
						# if no data for this gravity yet; make the whole dictionary!
						self.Q[self.g] = {}
						self.Q[self.g][s_old[0]] = {}
						self.Q[self.g][s_old[0]][s_old[1]] = {}
						self.Q[self.g][s_old[0]][s_old[1]][s_old[2]] = {}
						self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old] = (1-alpha)*Q0 + alpha*(self.last_reward + self.gamma*Qmax)      
		
		#print self.Q[self.g][s_old[0]][s_old[1]][s_old[2]][a_old]
		# update last action and state
		self.last_action = new_action
		self.last_state  = state

		# and return action
		return self.last_action

	def reward_callback(self, reward):
		'''This gets called so you can see what reward you get.'''
		self.last_reward = reward


def run_games(learner, hist, iters = 100, t_len = 1):
	'''
	Driver function to simulate learning by having the agent play a sequence of games.
	'''
	
	for ii in range(iters):
		# Make a new monkey object.
		swing = SwingyMonkey(sound=False,                  # Don't play sounds.
							 text="Epoch %d" % (ii),       # Display the epoch on screen.
							 tick_length = t_len,          # Make game ticks super fast.
							 action_callback=learner.action_callback,
							 reward_callback=learner.reward_callback)

		# Loop until you hit something.
		while swing.game_loop():
			pass
		
		# Save score history.
		hist.append(swing.score)
		#print swing.score, learner.eps

		# Reset the state of the learner.
		learner.reset()

		#print ii
		
	return


if __name__ == '__main__':

	# Select agent.
	agent = Learner()

	# Empty list to save history.
	hist = []

	# Run games. 
	run_games(agent, hist, 5000, 1)

	print max(hist)

	# Save history. 
	np.save('hist2',np.array(hist))


