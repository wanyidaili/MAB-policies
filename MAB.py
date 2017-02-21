# author: Wanyi Li
# OIT 674 homework assignment
# Febuary 2017
# Description: Consider a setting with two arms (k=2), 
# 	each arm with reward that is Bernoulli r.v., arm 1
# 	with mean 0.75 and arm 2 with mean 0.25. Your assignment 
#  	is to design and conduct a numerical experiment that 
#  	compares 4 MAB policies: 1. epsilon greedy; 2. upper confidence
#	bound; 3. Thompson Sampling; 4. Exp3. There are two objectives:
#	One: Compare the regret rate the policies achieve (if you want to 
#	argue a certain policy achieves logarithmic regret, you need to show 
#	it somehow based on your experiment), and also compare policies that
#	have the same regret rate (how would you determine which is better?)
#	Two: Repeat the experiment and analysis in a setting where the arms switch
#	their reward distribution in the middle of the horizon (arm 1 changes 
#  	from mean 0.75 to mean 0.25, and arm 2 switches from mean 0.25 to 
#	mean 0.75). This time, measure regret relative to the single best 
#	action in hindsight (this benchmark will be discussed on February 16th). 
# 	What is the performance gap between this benchmark and the best dynamic 
#  	sequence of actions?
################################################################################

import numpy as np;
import scipy as sp;
import matplotlib.pyplot as plt

####################################################
# Given arm i equals to 1 or 2, this function returns a Bernouli r.v.
# with either p=0.75 or p=0.25.
def Bern(i):
	if i == 1:
		rand = np.random.rand()
		return (1. if rand <=0.75 else 0.)
	if i == 2:
		rand = np.random.rand()
		return (1. if rand <=0.25 else 0.)

#########################################################
# Epsilon Greedy Method: pull every arm once; then at each t>k, select an arm
# uniformly with p = epsilon and choose the best arm so far with (1-p).
# Input T is the total time periods; e is the epsilon probability.
def eGreedy(T):
	# initialize accumulative reward of each arm
	x1, x2 = 0., 0.
	# initialize the number of times each arm has been pulled
	n1, n2 = 0., 0.
	def pull1():
		return n1 + 1, Bern(1) + x1
	def pull2():
		return n2 + 1, Bern(2) + x2

	e = 2*np.log(T)/T
	for i in range(T):
		# at first just pull each arm once
		if i == 0:
			n1, x1 = pull1()
		elif i == 1:
			n2, x2 = pull2()
		else:
			rand = np.random.rand()
			if rand <= e:
				arm = np.random.rand()
				if arm <= 0.5:
					n1, x1 = pull1()
				else:
					n2, x2 = pull2()
			else:
				if (x1/n1)>=(x2/n2):
					n1, x1 = pull1()
				else:
					n2, x2 = pull2()

	# calculate regret given the best arm's average is 0.75
	R = 0.75 * T - x1 - x2
	return R

#########################################
# Upper confidence bound method: pull each arm once, and then at every time step
# afterwards, play the arm that maximizes a quantity that is not fully myopic.
def UCB(T):
	# initialize accumulative reward of each arm
	x1, x2 = 0., 0.
	# initialize the number of times each arm has been pulled
	n1, n2 = 0., 0.
	def pull1():
		return n1 + 1, Bern(1) + x1
	def pull2():
		return n2 + 1, Bern(2) + x2

	for i in range(T):
		if i == 0:
			n1, x1 = pull1()
		elif i == 1:
			n2, x2 = pull2()
		else:
			M1 = x1/n1 + np.sqrt(2*np.log(T)/n1)
			M2 = x2/n2 + np.sqrt(2*np.log(T)/n2)
			if M1 >= M2:
				n1, x1 = pull1()
			else:
				n2, x2 = pull2()
	# compute regret
	R = 0.75 * T - x1 - x2
	return R

##################################################
# Thompson Sampling: for each time step: sample from a Beta distribution and then
# play the arm with the biggest sampling; then update the beta parameters.
def ThomSam(T):
	# initialize accumulative reward of each arm
	x1, x2 = 0., 0.
	# initialize the number of times each arm has been pulled
	n1, n2 = 0., 0.
	def pull1():
		success = Bern(1)
		return n1 + 1, success + x1, success
	def pull2():
		success = Bern(2)
		return n2 + 1, success + x2, success

	# initialize successes and failures for each arm
	s1, s2, f1, f2 = 0., 0., 0., 0.
	for i in range(T):
		theta1 = np.random.beta(s1+1, f1+1)
		theta2 = np.random.beta(s2+1, f2+1)
		if theta1 >= theta2:
			n1, x1, success = pull1()
			if success == 1:
				s1 += 1
			else:
				f1 += 1
		else:
			n2, x2, success = pull2()
			if success == 1:
				s2 += 1
			else:
				f2 += 1

	# compute regret
	R = 0.75 * T - x1 - x2
	return R


############################################################################
# EXP3 - Exponential Algorithm for Exploration and Exploitation: compute a probability
# distribution and pull arms from that distrubution. Observe the reward and update the
# estimated reward for each arm so that we can update the probability distrubition as
# well. This probabity to draw each arm is based on something similar to epsilon
# greedy as in with gamma probability it is a uniform random draw, the rest comes from
# the exponential weight parameter.
def EXP3(T):
	# initialize accumulative reward of each arm
	x1, x2 = 0., 0.
	# initialize the number of times each arm has been pulled
	n1, n2 = 0., 0.
	def pull1():
		success = Bern(1)
		return n1 + 1, success + x1, success
	def pull2():
		success = Bern(2)
		return n2 + 1, success + x2, success

	# compute gamma using Corrolary 3.2 in the paper Auer et al. 2002
	gamma = min(1, np.sqrt(2*np.log(2)/((np.exp(1)-1)*0.75*T)) )
	w = np.ones(2) # weight vector with lenth of the number of arms
	xtilda = np.zeros(2) # reward estimator
	for i in range(T):	
		wsum = sum(w)
		p1 = (1 - gamma)*w[0]/wsum + gamma/2
		p2 = (1 - gamma)*w[1]/wsum + gamma/2 # here p1 = 1 - p2
		rand = np.random.rand()
		if rand < p1:
			n1, x1, success = pull1()
			xtilda[0] = success/p1
			xtilda[1] = 0
			
		else:
			n2, x2, success = pull2()
			xtilda[1] = success/p2
			xtilda[0] = 0
		w[0] = w[0] * np.exp(gamma*xtilda[0]/2)
		w[1] = w[1] * np.exp(gamma*xtilda[1]/2)
	# compute (WORST CASE??? but still needs the 0.75 constant) regret
	R = 0.75*T - x1 - x2
	return R




########## TESTING SPACE ############################
if __name__ == '__main__':
	ts = []
	r2s = []
	r3s = []
	r1s = []

	# for t in np.logspace(2,4,50):
	# 	t = int(t)
	# 	r1 = []
	# 	r2 = []
	# 	r3 = []
	# 	for j in range(500):
	# 		r1.append(eGreedy(t))
	# 		r2.append(UCB(t))
	# 		r3.append(ThomSam(t))
			
	# 	ts.append(t)
	# 	r1s.append(np.average(r1))
	# 	r2s.append(np.average(r2))
	# 	r3s.append(np.average(r3))
		

	# r = plt.semilogx(ts,r2s, 'ro', label = 'UCB1') #UCB
	# b = plt.semilogx(ts,r1s, 'bo',label = 'egreedy') #egreedy
	# y = plt.semilogx(ts,r3s, 'yo',label='Thompson Sampling') #thomsam
	
	# #plt.title('UCB1')
	# plt.xlabel('time horizon')
	# plt.ylabel('regret')
	# plt.legend(
 #           scatterpoints=1,
 #           loc='upper left',
 #           ncol=3,
 #           fontsize=10)
	# plt.savefig('GeneralSemiLog1.jpg')
	r4s = []
	e4 = []
	for t in np.arange(100,10000,100):
		t = int(t)
		r1 = []
		r2 = []
		r3 = []
		r4 =[]
		for j in range(100):
			r4.append(EXP3(t))
			#r1.append(eGreedy(t))
			#r2.append(UCB(t))
			#r3.append(ThomSam(t))
		ts.append(t)
		#r1s.append(np.average(r1))
		#r2s.append(np.average(r2))
		#r3s.append(np.average(r3))
		r4s.append(np.average(r4))
		e4.append(np.sqrt(2*t))
	#plt.plot(ts,r1s, 'bo',label='eGreedy') # eGreedy
	#plt.plot(ts,r2s, 'ro',label='UCB1') # UCB1
	#plt.plot(ts,r3s, 'yo',label='Thompson Sampling') # ThomSam
	plt.plot(ts,r4s, 'go',label='EXP3') # EXP3
	plt.plot(ts, e4, '-',label='Expected: sqrt(2T)')
	plt.xlabel('time horizon')
	plt.ylabel('regret')
	plt.legend(
           scatterpoints=1,
           loc='upper left',
           ncol=3,
           fontsize=10)
	plt.savefig('EXP3_trial.jpg')
