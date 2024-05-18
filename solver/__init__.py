import numpy as np
import copy
import sys

"""
class : Richards式を解くためのクラス
att :
	voxel -> <np array> (Nx, Ny, Nz)なshape。if voxel[i,j,k] == 0 -> voidセル(石とか)。
	topNode -> <np array> 上面セル
	bottomNode -> <np array> 底面セル
	size -> <tuple> セルサイズ
	shape -> <tuple> 格子数
	h -> <np array> マトリックポテンシャル分布
	k -> <np array> 飽和透過係数分布
	theta_s -> <np arrar> 最大含水率分布
	theta_r -> <np array> 最小含水率分布
	alpha, n, m, l -> <np array> van Genuchtenモデルのパラメータ分布
	B -> <np array> 根密度分布
	a -> <np array> Feddesの式のパラメータ分布
	dead_flag -> <bool> 計算エラーのときTrue
"""
class field:
	"""
	Note :
	-- m is None -> m = 1-(1/n) * van Genuchtenでよく用いられる値
	-- l is None -> l = 0.5 * van Genuchtenでよく用いられる値
	-- B is None -> 植物モデル未考慮
	-- a is None -> 植物モデル未考慮
	"""
	def __init__(self, voxel, topNode, bottomNode, size, h, k, theta_s, theta_r, alpha, n, m = None, l = None, B = None, a = None, h50 = None, p = None):
		self.voxel = voxel
		self.topNode = topNode
		self.bottomNode = bottomNode
		self.size = size
		self.shape = voxel.shape
		self.dead_flag = False

		##########マトリックポテンシャルら物理場を定義。voxel == 0 -> 物理場の値はnp.nan
		self.h = np.where(self.voxel, h, np.nan)
		self.k = np.where(self.voxel, k, np.nan)
		self.theta_s = np.where(self.voxel, theta_s, np.nan)
		self.theta_r = np.where(self.voxel, theta_r, np.nan)
		self.alpha = np.where(self.voxel, alpha, np.nan)
		self.n = np.where(self.voxel, n, np.nan)
		self.m = np.where(self.voxel, 1.-1./n, np.nan) if (m is None) else np.where(self.voxel, m, np.nan)
		self.l = np.where(self.voxel, 0.5, np.nan) if (l is None) else np.where(self.voxel, l, np.nan)
		self.B = None if (B is None) else np.where(self.voxel, B, np.nan)

		if a is None:
			self.a0 = None; self.a1 = None; self.a2 = None; self.a3 = None
		else:
			a0 = a[:,:,:,0]; a1 = a[:,:,:,1]; a2 = a[:,:,:,2]; a3 = a[:,:,:,3]
			self.a0 = np.where(self.voxel, a0, np.nan); self.a1 = np.where(self.voxel, a1, np.nan)
			self.a2 = np.where(self.voxel, a2, np.nan); self.a3 = np.where(self.voxel, a3, np.nan)

		self.h50 = None if (h50 is None) else np.where(self.voxel, h50, np.nan)
		self.p = None if (p is None) else np.where(self.voxel, p, np.nan)


	"""
	process : fieldクラスのコピーを作成
	output : <field class>
	"""
	def copy(self):
		voxel = copy.deepcopy(self.voxel); topNode = copy.deepcopy(self.topNode); bottomNode = copy.deepcopy(self.bottomNode)
		size = copy.deepcopy(self.size); h = copy.deepcopy(self.h); k = copy.deepcopy(self.k)
		theta_s = copy.deepcopy(self.theta_s); theta_r = copy.deepcopy(self.theta_r); alpha = copy.deepcopy(self.alpha)
		n = copy.deepcopy(self.n); m = copy.deepcopy(self.m); l = copy.deepcopy(self.l); B = copy.deepcopy(self.B)
		a = None if (self.a0 is None) else copy.deepcopy(np.stack((self.a0, self.a1, self.a2, self.a3), axis = -1))

		return type(self)(voxel, topNode, bottomNode, size, h, k, theta_s, theta_r, alpha, n, m, l, B, a)

	def getH(self, ghost = np.nan):
		return np.where(self.voxel, self.h, ghost)

	def getSe(self, ghost = np.nan):
		return np.where(self.voxel, vanGenuchten_Se(self.h, self.alpha, self.n, self.m), ghost)
	
	def getK(self, ghost = np.nan):
		return np.where(self.voxel, vanGenuchten_K(self.h, self.k, self.alpha, self.n, self.m, self.l), ghost)
	
	def getCw(self, ghost = np.nan):
		return np.where(self.voxel, vanGenuchten_Cw(self.alpha, self.n, self.theta_s, self.theta_r, self.h), ghost)
	
	def getTheta(self, ghost = np.nan):
		return np.where(self.voxel, vanGenuchten_Theta(self.h, self.alpha, self.n, self.m, self.theta_s, self.theta_r), ghost)
	
	"""
	/*******************/
	process : ソース項の計算
	/*******************/
	input : Tp -> <np:float:(Nx, Ny)> 蒸散分布 [m/s]
	"""
	def getS(self, Tp = None, ghost = np.nan):
		if Tp is None:
			return np.where(self.voxel, 0., ghost)
		else:
			if self.a0 is None:
				F = np.where(self.voxel, S_Shaped(self.h, self.h50, self.p), ghost)
			else:
				F = np.where(self.voxel, Feddes(self.h, self.a0, self.a1, self.a2, self.a3), ghost)
			
			Tp = np.stack([Tp]*self.shape[2], axis = -1)
			return -F*Tp*self.B

"""
process : van Genuchtenモデルに従い、実飽和率[-]を計算。
input : h, alpha, n, m
	h -> <float> マトリックポテンシャル[m]。
	alpha, n, m -> <float> van Genuchtenモデルのパラメータ。
output : <float> 実飽和率[-]
"""
def vanGenuchten_Se(h, alpha, n, m):
	return (1.+abs(alpha*h)**n)**(-m)

def vanGenuchten_Theta(h, alpha, n, m, theta_s, theta_r):
	Se = vanGenuchten_Se(h, alpha, n, m)
	return (theta_s-theta_r)*Se+theta_r

def vanGenuchten_h(theta, alpha, n, m, theta_s, theta_r):
	Se = (theta-theta_r)/(theta_s-theta_r)
	if Se == 1.:
		return 0.
	elif Se == 0.:
		return -1e+10
	else:
		return -((Se**(-1./m)-1.)**(1./n))/alpha

"""
process : van Genuchtenモデルに従い、透過率[m/s]を計算。
input : h, k, alpha, n, m, l
	h -> <float> マトリックポテンシャル[m]。
	k -> <float> 飽和透過率 [m/s]
	alpha, n, m, l -> <float> van Genuchtenモデルのパラメータ。
output : <float> 実飽和率[-]
"""
def vanGenuchten_K(h, k, alpha, n, m, l):
	Se = vanGenuchten_Se(h, alpha, n, m) #<float> 実飽和率[-]
	
	return k*(Se**l)*(1.-(1.-Se**(1./m))**m)**2

"""
process : van Genuchtenモデルに従い、水分容量を計算。
input : h, k, alpha, n, m, l
	h -> <float> マトリックポテンシャル[m]。
	k -> <float> 飽和透過率 [m/s]
	alpha, n, m, l -> <float> van Genuchtenモデルのパラメータ。
output : <float> 実飽和率[-]
"""
def vanGenuchten_Cw(alpha,  n, theta_s, theta_r, h):
	Cw = (alpha**n)*(theta_s-theta_r)*(n-1.)*((-h)**(n-1.))
	Cw /= (1.+((-alpha*h)**n))**(2.-1./n)

	return Cw

"""
process : Feddesの式における係数alphaを計算。
input : h, a
	h -> <float> マトリックポテンシャル[m]。
	a* -> <float> 折線の節点
output : <float> パラメータalpha
"""
def Feddes(h, a0, a1, a2, a3):
	region2 = (h>a1)*(h<a0)
	region3 = (h<=a1)*(h>=a2)
	region4 = (h>a3)*(h<a2)

	F = region2*(a0-h)/(a0-a1)+region3+region4*(h-a3)/(a2-a3)

	return F

"""
process : S_shaped functionにおける係数alphaを計算。
input : h, h50, p
	h -> <float> マトリックポテンシャル[m]。
	h50 -> <float> alphaが0.5になるマトリックポテンシャル [m]
	p -> <float> 係数
output : <float> パラメータalpha
"""
def S_Shaped(h, h50, p):
	return 1./(1.+np.abs(h/h50)**p)