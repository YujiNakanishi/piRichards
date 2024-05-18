import numpy as np
import sys

"""
process : Carselの分布に従い、Sand質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSand(num):
	mu = np.array([-0.394, -3.12, 0.378, 0.978])
	T = np.array([
		[1.04, 0., 0., 0.],
		[-0.109, 0.182, 0., 0.],
		[0.328, 0.258, 0.143, 0.],
		[0.081, -0.047, -0.011, 0.017]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = 70.*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = np.exp(y[:,1])
		alpha = 0.25*np.exp(y[:,2])/(1.+np.exp(y[:,2]))
		n = np.exp(y[:,3])

		limit_mask = (0. < Ks)*(Ks < 70.)*(0. < thetar)*(thetar < 0.1)*(0. < alpha)*(alpha < 0.25)*(1.5 < n)*(n < 4.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Sandy Loam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSandyLoam(num):
	mu = np.array([-2.49, 0.384, -0.937, 0.634])
	T = np.array([
		[1.6, 0., 0., 0.],
		[-0.153, 0.538, 0., 0.],
		[0.037, 0.017, 0.014, 0.],
		[0.211, -0.194, 0.019, 0.108]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = 30.*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = 0.11*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = 0.25*np.exp(y[:,2])/(1.+np.exp(y[:,2]))
		n = np.exp(y[:,3])

		limit_mask = (0. < Ks)*(Ks < 30.)*(0. < thetar)*(thetar < 0.11)*(0. < alpha)*(alpha < 0.25)*(1.35 < n)*(n < 3.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample

"""
process : Carselの分布に従い、Loamy Sand質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleLoamySand(num):
	mu = np.array([-1.27, 0.075, 0.124, -1.11])
	T = np.array([
		[1.48, 0., 0., 0.],
		[-0.201, 0.522, 0., 0.],
		[0.037, 0.017, 0.014, 0.],
		[0.211, -0.194, 0.019, 0.108]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = 51.*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = 0.11*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = y[:,2]
		n = (5.*np.exp(y[:,3]) + 1.35) / (1. + np.exp(y[:,3]))

		limit_mask = (0. < Ks)*(Ks < 51.)*(0. < thetar)*(thetar < 0.11)*(0. < alpha)*(alpha < 0.25)*(1.35 < n)*(n < 5.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、SiltLoam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSiltLoam(num):
	mu = np.array([-2.19, 0.478, -4.1, -0.37])
	T = np.array([
		[1.478, 0., 0., 0.],
		[-0.201, 0.522, 0., 0.],
		[0.525, 0.03, 0.082, 0.],
		[0.353, -0.17, 0.234, 0.158]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = np.exp(y[:,0])
		thetar = 0.11*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = np.exp(y[:,2])
		n = (2.*np.exp(y[:,3]) + 1.) / (1. + np.exp(y[:,3]))

		limit_mask = (0. < Ks)*(Ks < 15.)*(0. < thetar)*(thetar < 0.11)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 2.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Silt質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSilt(num):
	mu = np.array([-2.2, 0.042, 0.017, 1.38])
	T = np.array([
		[0.535, 0., 0., 0.],
		[-0.002, 0.008, 0., 0.],
		[0.003, 0., 0.001, 0.],
		[0.013, -0.015, 0.014, 0.013]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		trunction_mask = (-2.564 < y[:,0])*(y[:,0] < -0.337)*(0.013 < y[:,1])*(y[:,1] < 0.049)
		y = y[trunction_mask]

		Ks = np.exp(y[:,0])
		thetar = y[:,1]
		alpha = y[:,2]
		n = y[:,3]

		limit_mask = (0. < Ks)*(Ks < 2.)*(0. < thetar)*(thetar < 0.09)*(0. < alpha)*(alpha < 0.1)*(1.2 < n)*(n < 1.6)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Clay質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleClay(num):
	mu = np.array([-5.75, 0.445, -4.145, 0.0002])
	T = np.array([
		[1.96, 0., 0., 0.],
		[0.07, 0.017, 0., 0.],
		[0.565, -0.08, 0.172, 0.],
		[0.048, -0.014, 0.002, 0.016]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		trunction_mask = (0.0065 < y[:,1])*(y[:,1] < 0.834)*(-5.01 < y[:,2])*(y[:,2] < 0.912)*(0. < y[:,3])*(y[:,3] < 0.315)
		y = y[trunction_mask]

		Ks = 5.*np.exp(y[:,0])/(1. + np.exp(y[:,0]))
		thetar = 0.15*(np.exp(y[:,1])-np.exp(-y[:,1]))/2.
		alpha = 0.15*np.exp(y[:,2])/(1. + np.exp(y[:,2]))
		n = np.exp(y[:,3])

		limit_mask = (0. < Ks)*(Ks < 5.)*(0. < thetar)*(thetar < 0.15)*(0. < alpha)*(alpha < 0.15)*(0.9 < n)*(n < 1.4)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Silty Clay質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSiltyClay(num):
	mu = np.array([-5.69, 0.07, -5.66, -1.28])
	T = np.array([
		[1.25, 0., 0., 0.],
		[0.008, 0.003, 0., 0.],
		[0.314, 0.04, 0.06, 0.],
		[0.367, -0.086, 0.066, 0.131]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = np.exp(y[:,0])
		thetar = y[:,1]
		alpha = np.exp(y[:,2])
		n = (1.4*np.exp(y[:,3])+1.)/(1.+np.exp(y[:,3]))

		limit_mask = (0. < Ks)*(Ks < 1.)*(0. < thetar)*(thetar < 0.14)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 1.4)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Sandy Clay質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSandyClay(num):
	mu = np.array([-4.04, 1.72, -3.77, 0.202])
	T = np.array([
		[2.02, 0., 0., 0.],
		[0.883, 0.324, 0., 0.],
		[0.539, 0.063, 0.15, 0.],
		[0.076, 0.004, -0.001, 0.018]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = np.exp(y[:,0])
		thetar = 0.12*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = np.exp(y[:,2])
		n = np.exp(y[:,3])

		limit_mask = (0. < Ks)*(Ks < 1.5)*(0. < thetar)*(thetar < 0.12)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 1.5)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Silty Clay Loam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSiltyClayLoam(num):
	mu = np.array([-5.31, 0.088, -2.75, 1.23])
	T = np.array([
		[1.612, 0., 0., 0.],
		[0.006, 0.005, 0., 0.],
		[0.511, 0.048, 0.073, 0.],
		[0.049, -0.009, 0.008, 0.017]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = 3.5*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = y[:,1]
		alpha = 0.15*np.exp(y[:,2])/(1.+np.exp(y[:,2]))
		n = y[:,3]

		limit_mask = (0. < Ks)*(Ks < 3.5)*(0. < thetar)*(thetar < 0.115)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 1.5)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Clay Loam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleClayLoam(num):
	mu = np.array([-5.87, 0.679, -4.22, 0.132])
	T = np.array([
		[1.92, 0., 0., 0.],
		[0.04, 0.031, 0., 0.],
		[0.589, -0.062, 0.106, 0.],
		[0.542, -0.154, 0.065, 0.116]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		trunction_mask = (-8.92 < y[:,0])*(y[:,0] < 2.)
		y = y[trunction_mask]

		Ks = 7.5*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = 0.13*(np.exp(y[:,1])-np.exp(-y[:,1]))/2.
		alpha = np.exp(y[:,2])
		n = (1.6*np.exp(y[:,0])+1.)/(1.+np.exp(y[:,0]))

		limit_mask = (0. < Ks)*(Ks < 7.5)*(0. < thetar)*(thetar < 0.13)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 1.6)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample



"""
process : Carselの分布に従い、Sandy Clay Loam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleSandyClayLoam(num):
	mu = np.array([-4.04, 1.65, -1.38, 0.388])
	T = np.array([
		[1.85, 0., 0., 0.],
		[0.102, 0.378, 0., 0.],
		[0.784, 0.122, 0.22, 0.],
		[0.077, -0.031, -0.008, 0.016]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		trunction_mask = (0.928 < y[:,1])*(y[:,1] < 2.94)
		y = y[trunction_mask]

		Ks = 20.*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = 0.12*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = 0.25*np.exp(y[:,2])/(1.+np.exp(y[:,2]))
		n = np.exp(y[:,3])

		limit_mask = (0. < Ks)*(Ks < 20.)*(0. < thetar)*(thetar < 0.12)*(0. < alpha)*(alpha < 0.25)*(1. < n)*(n < 2.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample


"""
process : Carselの分布に従い、Loam質の物性値を返す。
input : num -> <int> サンプル数
output : sample -> <np array> (num, 4)
	Ks, theta_r, alpha, nを格納 *単位はSIに従う。
"""
def sampleLoam(num):
	mu = np.array([-3.71, 0.639, -1.27, 0.532])
	T = np.array([
		[1.41, 0., 0., 0.],
		[-0.1, 0.478, 0., 0.],
		[0.611, 0.073, 0.093, 0.],
		[0.055, -0.055, 0.026, 0.029]])

	sample = np.ones((0, 4))

	while len(sample) < num:
		res_num = num-len(sample)
		z = np.random.rand(res_num, 4)
		y = mu + z@T

		Ks = 15.*np.exp(y[:,0])/(1.+np.exp(y[:,0]))
		thetar = 0.12*np.exp(y[:,1])/(1.+np.exp(y[:,1]))
		alpha = 0.15*np.exp(y[:,2])/(1.+np.exp(y[:,2]))
		n = 1.+(np.exp(y[:,3])-np.exp(-y[:,3]))/2.

		limit_mask = (0. < Ks)*(Ks < 15.)*(0. < thetar)*(thetar < 0.12)*(0. < alpha)*(alpha < 0.15)*(1. < n)*(n < 2.)
		sample = np.concatenate((sample, np.stack((Ks, thetar, alpha, n), axis = 1)[limit_mask]), axis = 0)

	sample[:,0] /= (100.*60.*60.); sample[:,2] *= 100.

	return sample