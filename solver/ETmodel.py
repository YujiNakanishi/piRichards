import numpy as np
import math
import copy

"""
/***************/
process : エネルギー単位を [MJ/m2/day]から[mm/day]に変換
/***************/
input :
	E -> <float> エネルギー [MJ/m2/day]
output: <float> エネルギー [mm/day]
"""
def MJ2MM(E):
	return 0.408*E

"""
/***************/
process : psychrometric constant [kPa/degC]を計算
/***************/
input :
	P -> <float> 気圧 [kPa]
output: <float> gamma [kPa/degC]
"""
def getGamma(P):
	return (6.65e-4)*P

"""
/***************/
process : 飽和蒸気圧 [kPa]を計算
/***************/
input :
	T -> <float> 気温 [degC]
output: <float> 飽和蒸気圧 [kPa]
"""
def get_e0(T):
	return 0.6108*math.exp(17.27*T/(T+237.3))

"""
/***************/
process : 蒸気圧 [kPa]を計算
/***************/
input :
	T -> <float> 気温 [degC]
	RH -> <float> 相対湿度 [%]
output: <float> 蒸気圧 [kPa]
"""
def get_ea(T, RH):
	return get_e0(T)*(RH/100.)

"""
/***************/
process : 飽和蒸気圧の勾配 [kPa/degC]を計算
/***************/
input :
	T -> <float> 気温 [degC]
output: <float> 飽和蒸気圧の勾配 [kPa/degC]
"""
def getDelta(T):
	return 2503.*math.exp(17.27*T/(T+237.3))/((T+237.3)**2)

"""
/***************/
process : 地上z mの風速から2 mでの風速を予測
/***************/
input :
	z -> <float> 高さ [m]
	uz -> <float> 風速 [m/s]
output: <float> u2 [m/s]
"""
def convert_u(uz, z):
	return 4.87*uz/math.log(67.8*z-5.42)

"""
/********************/
process : net radiation[mm/day]の計算
/********************/
input :
	J -> <float> 1月1日から計算した経過日数 (1 <= J <= 365 or 366)
	lattitude -> <float> 緯度[deg] (-90 <= lattitude <= 90)
	n -> <float> 日照時間 [h]
	ea -> <float> 蒸気圧 [kPa]
	Tmax -> <float> 1日の最高気温 [degC]
	Tmin -> <float> 1日の最低気温 [degC]

output:
	Rn -> <float> [mm/day]
"""
def getRn(J, lattitude, n, ea, Tmax, Tmin):
	dr = 1.+0.033*math.cos(2.*np.pi/365.*J) #inverse relative distance Earth-Sun
	delta = 0.409*math.sin(2.*np.pi/365.*J - 1.39) #solar declination [rad]
	varphi = lattitude*np.pi/180. #[rad]
	omegas = math.acos(-math.tan(varphi)*math.tan(delta))

	Ra = 0.082*(24.*60./np.pi)*dr*(omegas*math.sin(varphi)*math.sin(delta)+math.cos(varphi)*math.cos(delta)*math.sin(omegas)) #[MJ/m2/day]
	Ra = MJ2MM(Ra) #[mm/day]

	N = 24.*omegas/np.pi

	Rs = (0.25 + 0.5*n/N)*Ra #[mm/day]
	Rns = (1.-0.23)*Rs #[mm/day]
	Rso = 0.75*Ra #[mm/day]

	Tmean4 = ((Tmax+273.)**4)+((Tmin+273.)**4)
	Rnl = (4.903e-9)*Tmean4*(0.34-0.14*math.sqrt(ea))*(1.35*Rs/Rso-0.35) #[MJ/m2/day]
	Rnl = MJ2MM(Rnl)

	return Rns - Rnl


"""
/********************/
process : G[mm/day]の計算
/********************/
input :
	Rn -> <float> net radiation [mm/day]
	time -> <str> "daylight" or "nighttime"
output:
	G -> <float> [mm/day]
"""
def getG(Rn, time):
	if time == "daylight":
		return 0.1*Rn
	else:
		return 0.5*Rn


"""
/********************/
class : FAO Penman_Monteithモデルと植物係数に関するクラス
/********************/
att:
	L_ini, L_dev, L_mid, L_late -> <float> 作物ステージ日数(FAO Table 11) [day]
	h -> <float> 植物高さ [m]
	Kc -> <float> Kc値 (生育ステージ未考慮の場合)
	Kc_ini, Kc_mid, Kc_end -> <float> Kc値(生育ステージ考慮の場合)(FAO Table 12)
	LAI -> <np:float:(Nx, Ny)> LAI分布
	constant -> <bool> 生育ステージ考慮か否か。考慮 -> constant = False
"""
class ETcModule:
	def __init__(self, Kc, h, LAI, L = None):
		self.h = h
		self.LAI = LAI

		if L is None:
			self.Kc = Kc
			self.constant = True
		else:
			self.L_ini = L[0]; self.L_dev = L[1]; self.L_mid = L[2]; self.L_late = L[3]
			self.Kc_ini = Kc[0]; self.Kc_mid = Kc[1]; self.Kc_end = Kc[2]
			self.constant = False

	"""
	/***************/
	process : ET0値が既知の場合の、蒸発および蒸散計算
	/***************/
	input :
		ET0 -> <np:float:(Nx, Ny)> ET0値 [m/s]
		u2 -> <float> 地上2 mでの風速 [m/s]
		T -> <float> 気温 [degC]
		RHmin -> <float> 日の最低湿度 [%]
		L -> <int> 作物栽培経過日数 [day] * None -> 作物の生育ステージ未考慮。
	output:
		E -> <np:float:(Nx, Ny)> 蒸発量 [m/s]
		Tp -> <np:float:(Nx, Ny)> 蒸散量 [m/s]
	"""
	def __call__(self, ET0, u2 = 2., RHmin = 45., L = None):
		ETc = self.getKc(u2, RHmin, L)*ET0 #<float> [m/s]
		return self.Campbell(ETc)


	"""
	/***************/
	process : FAO Penman Monteith法とCampbellの式による蒸発、蒸散の計算
	/***************/
	input :
		Delta -> <float> 蒸気圧曲線の勾配 [kPa/degC]
		Rn -> <float> net radiation [mm/day]
		G -> <float> soil heat flux [mm/day]
		es -> <float> 飽和蒸気圧 [kPa]
		ea -> <float> 蒸気圧 [kPa]
		gamma -> <float> psychrometric constant [kPa/degC]
		u2 -> <float> 地上2 mでの風速 [m/s]
		T -> <float> 気温 [degC]
		RHmin -> <float> 日の最低湿度 [%]
		L -> <int> 作物栽培経過日数 [day] * None -> 作物の生育ステージ未考慮。
	output:
		E -> <np:float:(Nx, Ny)> 蒸発量 [m/s]
		Tp -> <np:float:(Nx, Ny)> 蒸散量 [m/s]
	"""
	def FAO_Penman_Monteith(self, Delta, Rn, es, ea, gamma, T, G = 0., u2 = 2., RHmin = 45., L = None):
		ET0 = (Delta*(Rn-G)+900.*gamma*u2*(es-ea)/(T+273.))/(Delta+gamma*(1.+0.34*u2)) #<float> [mm/day]
		ETc = self.getKc(u2, RHmin, L)*ET0 #<float> [mm/day]
		ETc /= (1000.*24*60*60)

		return self.Campbell(ETc)

	def Campbell(self, ETc):
		Tp = ETc*(1.-np.exp(-0.463*self.LAI))
		E = ETc-Tp

		return E, Tp


	"""
	process : ETcModuleクラスのコピーを作成
	output : <ETcModule class>
	"""
	def copy(self):
		LAI = copy.deepcopy(self.LAI)

		if self.constant:
			return type(self)(self.Kc, self.h, LAI, None)
		else:
			self.L_ini = L[0]; self.L_dev = L[1]; self.L_mid = L[2]; self.L_late = L[3]
			self.Kc_ini = Kc[0]; self.Kc_mid = Kc[1]; self.Kc_end = Kc[2]

			return type(self)([self.Kc_ini, self.Kc_mid, self.Kc_end], self.h, LAI, [self.L_ini, self.L_dev, self.L_mid, self.L_late])

	"""
	/***************/
	process : 作物係数を計算
	/***************/
	input :
		u2 -> <float> 地上2 mでの風速 [m/s]
		T -> <float> 気温 [degC]
		RHmin -> <float> 日の最低湿度 [%]
		L -> <int> 作物栽培経過日数 [day] * None -> 作物の生育ステージ未考慮。
	output:
		Kc -> <float> 作物係数
	"""
	def getKc(self, u2, RHmin, L):
		if L is None:
			return self.Kc + (0.04*(u2-2.)-0.004*(RHmin-45.))*((self.h/3.)**0.3)
		else:
			if L < self.L_ini:
				return self.Kc_ini
			else:
				L -= self.L_ini
				#####Kcb_midの調整
				Kc_mid = self.Kc_mid + (0.04*(u2-2.)-0.004*(RHmin-45.))*((self.h/3.)**0.3)

				if L < self.L_dev:
					return self.Kc_ini + (Kc_mid-self.Kc_ini)*L/self.L_dev
				else:
					L -= self.L_dev
					if L < self.L_mid:
						return Kc_mid
					else:
						L -= self.L_mid
						Kc_end = self.Kc_end + (0.04*(u2-2.)-0.004*(RHmin-45.))*((self.h/3.)**0.3)

						if L > self.L_late:
							return Kc_end
						else:
							return Kc_mid + (Kc_end - Kc_mid)*L/self.L_late