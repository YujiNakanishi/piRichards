import numpy as np
import math
import sys

"""
process : Particle Filter
att : 
	inidividuals -> <list:Individual>
"""
class PF:
	def __init__(self, individuals):
		self.individuals = individuals

	def __len__(self):
		return len(self.individuals)

	"""
	process : 各個体の尤度を計算
	input :
		y -> <np array> 観測データ。 (N, )なshape
		R -> <np array> 観測データの分散共分散行列。(N, N)なshape
	"""
	def getProbs(self, y, R):
		likelihoods = np.array([individual.calcLikelihood(y, R) for individual in self.individuals])
		if np.sum(likelihoods) == 0.:
			return 1./len(self)*np.ones(len(self))
		else:
			likelihoods = np.array([individual.calcLikelihood(y, R) for individual in self.individuals])
			return likelihoods/np.sum(likelihoods)

	"""
	process : 各個体の尤度を基にサンプリング
	input :
		y -> <np array> 観測データ
		R -> <np array> 観測データの分散共分散行列
	"""
	def sampling(self, y, R):
		prob = self.getProbs(y, R)
		sample_index = np.random.choice(np.arange(len(self)), size = len(self), p = prob)

		self.individuals = [self.individuals[si].copy() for si in sample_index]

	"""
	process : 個体群の平均を返す
	output : <Individual class>
	"""
	def mean(self):
		mean_individual = self.individuals[0].truediv(len(self))
		for individual in self.individuals[1:]:
			mean_individual += individual.truediv(len(self))

		return mean_individual

	"""
	process : 個体群の分散を返す
	output : <Individual class>
	"""
	def var(self):
		mean_individual = self.mean()
		var_individual = (self.individuals[0] - mean_individual)*(self.individuals[0] - mean_individual).truediv(len(self))
		for individual in self.individuals[1:]:
			var_individual += (individual - mean_individual)*(individual - mean_individual).truediv(len(self))

		return var_individual

	"""
	process : 個体群のobserve結果の平均を返す
	output : <np array>
	"""
	def observe_mean(self):
		obsv = [individual.observe() for individual in self.individuals]
		return np.mean(obsv, axis = 0)

	"""
	process : 個体群のobserve結果の分散を返す
	output : <np array>
	"""
	def observe_var(self):
		obsv = [individual.observe() for individual in self.individuals]
		obsv_mean = np.mean(obsv, axis = 0)
		return np.mean([np.power(ob-obsv_mean, 2.) for ob in obsv], axis = 0)

"""
process : Merging Particle Filter
att : 
	inidividuals -> <list:Individual>
	a -> <np:float:(3,)> 重み。
"""
class MPF(PF):
	def __init__(self, individuals, a = None):
		super().__init__(individuals)
		self.a = np.array([3./4., (math.sqrt(13.)+1.)/8., -(math.sqrt(13.)-1.)/8.]) if (a is None) else a

	def sampling(self, y, R):
		prob = self.getProbs(y, R)
		sample_index = np.random.choice(np.arange(len(self)), size = 3*len(self), p = prob)

		individuals = np.array([self.individuals[si].copy() for si in sample_index]).reshape((-1, 3))
		individuals1 = [individual.mul(self.a[0]) for individual in individuals[:,0]]
		individuals2 = [individual.mul(self.a[1]) for individual in individuals[:,1]]
		individuals3 = [individual.mul(self.a[2]) for individual in individuals[:,2]]

		self.individuals = []

		for ind1, ind2, ind3 in zip(individuals1, individuals2, individuals3):
			ind = ind1+ind2+ind3
			ind.field.h[ind.field.h > 0.] = 0.

			self.individuals.append(ind)


"""
process : BLX_alpha
att : 
	inidividuals -> <list:Individual>
	alpha -> <float> 重み。
Note:
---個体数について---
個体数は偶数でなければならない。
"""
class BLX_alpha(PF):
	def __init__(self, individuals, alpha = 0.5):
		super().__init__(individuals)
		self.alpha = alpha

	def sampling(self, y, R):
		prob = self.getProbs(y, R)
		sample_index = np.random.choice(np.arange(len(self)), size = len(self), p = prob)
		individuals = np.array([self.individuals[si].copy() for si in sample_index]).reshape((-1, 2))
		self.individuals = []

		for i in range(len(individuals)):
			param1 = individuals[i,0].params
			param2 = individuals[i,1].params
			h1 = individuals[i,0].field.getH()
			h2 = individuals[i,1].field.getH()

			d = np.abs(param1 - param2) #(D, )

			param_mean = 0.5*(param1 + param2) #(D, )
			param_max = param_mean + (0.5 + self.alpha)*d
			param_min = param_mean - (0.5 + self.alpha)*d

			new_param1 = param_min + (param_max - param_min)*np.random.rand(len(d))
			new_param2 = param_min + (param_max - param_min)*np.random.rand(len(d))

			dh = np.abs(h1-h2)
			h_mean = 0.5*(h1 + h2)
			h_max = h_mean + (0.5 + self.alpha)*dh
			h_min = h_mean - (0.5 + self.alpha)*dh
			new_h1 = h_min + (h_max - h_min)*np.random.rand(dh.shape[0], dh.shape[1], dh.shape[2])
			new_h2 = h_min + (h_max - h_min)*np.random.rand(dh.shape[0], dh.shape[1], dh.shape[2])

			new_ind1 = type(individuals[i,0])(new_param1)
			new_ind1.createField(new_h1)
			new_ind2 = type(individuals[i,1])(new_param2)
			new_ind2.createField(new_h2)

			new_ind1.field.h[new_ind1.field.h > 0.] = 0.
			new_ind2.field.h[new_ind2.field.h > 0.] = 0.

			self.individuals.append(new_ind1); self.individuals.append(new_ind2)

"""
process : BLX_alpha_withoutH。ただしhの交叉はなし。
att : 
	inidividuals -> <list of Individual>
	alpha -> <float> 重み。
Note:
---個体数について---
個体数は偶数でなければならない。
"""
class BLX_alpha_withoutH(BLX_alpha):
	def sampling(self, y, R):
		prob = self.getProbs(y, R)
		sample_index = np.random.choice(np.arange(len(self)), size = len(self), p = prob)
		individuals = np.array([self.individuals[si].copy() for si in sample_index]).reshape((-1, 2))
		self.individuals = []

		for i in range(len(individuals)):
			param1 = individuals[i,0].params
			param2 = individuals[i,1].params
			h1 = individuals[i,0].field.getH() #hは片方の親の値を継承
			h2 = individuals[i,1].field.getH() #hは片方の親の値を継承

			d = np.abs(param1 - param2) #(D, )
			param_mean = 0.5*(param1 + param2) #(D, )
			param_max = param_mean + (0.5 + self.alpha)*d
			param_min = param_mean - (0.5 + self.alpha)*d

			new_param1 = param_min + (param_max - param_min)*np.random.rand(len(d))
			new_param2 = param_min + (param_max - param_min)*np.random.rand(len(d))

			new_ind1 = type(individuals[i,0])(new_param1)
			new_ind1.createField(h1)
			new_ind2 = type(individuals[i,1])(new_param2)
			new_ind2.createField(h2)

			self.individuals.append(new_ind1); self.individuals.append(new_ind2)