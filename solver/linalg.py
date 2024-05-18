import numpy as np

"""
process : ヤコビ法による定常解析
input :
	field -> <field class>
	q -> <ndarray> 地表面フラックス。(Nx, Ny)なshape
	top -> "zero" or "flux"
	bottom -> "free" or "zero"
	Tp -> <float> 蒸散量
	iteration -> <int> 反復回数
	lr -> <float> 緩和係数
output : なし。fieldのattが更新
Note :
-- 計算が発散した場合、例外処理が発動しfield.dead_flagがTrueになる
"""
def run_Steady(field, q = None, top = "flux", bottom = "free", Tp = None, iteration = 1000, lr = 0.9):
	dx, dy, dz = field.size #計算格子サイズ
	h_extend = np.ones((field.shape[0]+2, field.shape[1]+2, field.shape[2]+2))*np.nan
	K_extend = np.ones((field.shape[0]+2, field.shape[1]+2, field.shape[2]+2))*np.nan

	try:
		S = field.getS(Tp); K = field.getK()
		K_extend[1:-1,1:-1,1:-1] = K
		if top == "zero":
			K_extend[np.array(field.topNode[0])+1, np.array(field.topNode[1])+1, np.array(field.topNode[2])] = field.k[field.topNode[0], field.topNode[1], field.topNode[2]]

		if bottom == "zero":
			K_extend[np.array(field.bottomNode[0])+1, np.array(field.bottomNode[1])+1, np.array(field.bottomNode[2])] = field.k[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]]

		for itr in range(iteration):
			h_extend[1:-1,1:-1,1:-1] = field.h
			if top == "zero":
				h_extend[np.array(field.topNode[0])+1, np.array(field.topNode[1])+1, np.array(field.topNode[2])] = 0.	

			if bottom == "zero":
				h_extend[np.array(field.bottomNode[0])+1, np.array(field.bottomNode[1])+1, np.array(field.bottomNode[2])] = 0.
			
			K_right = (K_extend[2:,1:-1,1:-1]+K_extend[1:-1,1:-1,1:-1])/2. #右境界のK値
			a_right = K_right/(dx**2); a_right[np.isnan(a_right)] = 0. #右側係数
			s_right = a_right*h_extend[2:, 1:-1, 1:-1]; s_right[np.isnan(s_right)] = 0.

			K_left = (K_extend[:-2,1:-1,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_left = K_left/(dx**2); a_left[np.isnan(a_left)] = 0.
			s_left = a_left*h_extend[:-2, 1:-1, 1:-1]; s_left[np.isnan(s_left)] = 0.

			K_front = (K_extend[1:-1,2:,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_front = K_front/(dy**2); a_front[np.isnan(a_front)] = 0.
			s_front = a_front*h_extend[1:-1, 2:, 1:-1]; s_front[np.isnan(s_front)] = 0.

			K_back = (K_extend[1:-1,:-2,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_back = K_back/(dy**2); a_back[np.isnan(a_back)] = 0.
			s_back = a_back*h_extend[1:-1, :-2, 1:-1]; s_back[np.isnan(s_back)] = 0.

			K_up = (K_extend[1:-1,1:-1,2:]+K_extend[1:-1,1:-1,1:-1])/2.
			a_up = K_up/(dz**2); a_up[np.isnan(a_up)] = 0.
			s_up = a_up*h_extend[1:-1, 1:-1, 2:]; s_up[np.isnan(s_up)] = 0.
			b_up = K_up/dz; b_up[np.isnan(b_up)] = 0.
			if top == "flux":
				b_up[field.topNode[0], field.topNode[1], field.topNode[2]] = q[field.topNode[0], field.topNode[1]]/dz
			
			K_down = (K_extend[1:-1,1:-1,:-2]+K_extend[1:-1,1:-1,1:-1])/2.
			a_down = K_down/(dz**2); a_down[np.isnan(a_down)] = 0.
			s_down = a_down*h_extend[1:-1, 1:-1,:-2]; s_down[np.isnan(s_down)] = 0.
			b_down = -K_down/dz; b_down[np.isnan(b_down)] = 0.
			b_down[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]] = -K[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]]/dz

			a_i = a_right+a_left+a_front+a_back+a_up+a_down
			h_next = (s_right+s_left+s_front+s_back+s_up+s_down+S+b_up+b_down)/a_i
			field.h = (1.-lr)*field.h+lr*h_next
			field.h[field.h > 0] = 0.

	except:
		field.dead_flag = True

	if np.min(field.h) < -1e+100:
		field.dead_flag = True


"""
process : ヤコビ法による非定常解析
input :
	field -> <field class>
	dt -> <float> 時間刻み
	q -> <ndarray> 地表面フラックス。(Nx, Ny)なshape
	top -> "zero" or "flux"
	bottom -> "free" or "zero"
	Tp -> <float> 蒸散量
	iteration -> <int> 反復回数
	lr -> <float> 緩和係数
output : なし。fieldのattが更新
Note :
-- 計算が発散した場合、例外処理が発動しfield.dead_flagがTrueになる
"""
def run_Unsteady(field, dt, q = None, top = "flux", bottom = "free", Tp = None, iteration = 20, lr = 0.9):
	dx, dy, dz = field.size #計算格子サイズ
	h_before = field.h.copy() #現時刻のマトリックポテンシャル
	h_extend = np.ones((field.shape[0]+2, field.shape[1]+2, field.shape[2]+2))*np.nan
	K_extend = np.ones((field.shape[0]+2, field.shape[1]+2, field.shape[2]+2))*np.nan

	try:
		Cw = field.getCw(); S = field.getS(Tp); K = field.getK()
		K_extend[1:-1,1:-1,1:-1] = K
		if top == "zero":
			K_extend[np.array(field.topNode[0])+1, np.array(field.topNode[1])+1, np.array(field.topNode[2])] = field.k[field.topNode[0], field.topNode[1], field.topNode[2]]

		if bottom == "zero":
			K_extend[np.array(field.bottomNode[0])+1, np.array(field.bottomNode[1])+1, np.array(field.bottomNode[2])] = field.k[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]]

		for itr in range(iteration):
			h_extend[1:-1,1:-1,1:-1] = field.h
			if top == "zero":
				h_extend[np.array(field.topNode[0])+1, np.array(field.topNode[1])+1, np.array(field.topNode[2])] = 0.	

			if bottom == "zero":
				h_extend[np.array(field.bottomNode[0])+1, np.array(field.bottomNode[1])+1, np.array(field.bottomNode[2])] = 0.
			
			K_right = (K_extend[2:,1:-1,1:-1]+K_extend[1:-1,1:-1,1:-1])/2. #右境界のK値
			a_right = K_right/(dx**2); a_right[np.isnan(a_right)] = 0. #右側係数
			s_right = a_right*h_extend[2:, 1:-1, 1:-1]; s_right[np.isnan(s_right)] = 0.

			K_left = (K_extend[:-2,1:-1,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_left = K_left/(dx**2); a_left[np.isnan(a_left)] = 0.
			s_left = a_left*h_extend[:-2, 1:-1, 1:-1]; s_left[np.isnan(s_left)] = 0.

			K_front = (K_extend[1:-1,2:,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_front = K_front/(dy**2); a_front[np.isnan(a_front)] = 0.
			s_front = a_front*h_extend[1:-1, 2:, 1:-1]; s_front[np.isnan(s_front)] = 0.

			K_back = (K_extend[1:-1,:-2,1:-1]+K_extend[1:-1,1:-1,1:-1])/2.
			a_back = K_back/(dy**2); a_back[np.isnan(a_back)] = 0.
			s_back = a_back*h_extend[1:-1, :-2, 1:-1]; s_back[np.isnan(s_back)] = 0.

			K_up = (K_extend[1:-1,1:-1,2:]+K_extend[1:-1,1:-1,1:-1])/2.
			a_up = K_up/(dz**2); a_up[np.isnan(a_up)] = 0.
			s_up = a_up*h_extend[1:-1, 1:-1, 2:]; s_up[np.isnan(s_up)] = 0.
			b_up = K_up/dz; b_up[np.isnan(b_up)] = 0.
			if top == "flux":
				b_up[field.topNode[0], field.topNode[1], field.topNode[2]] = q[field.topNode[0], field.topNode[1]]/dz
			
			K_down = (K_extend[1:-1,1:-1,:-2]+K_extend[1:-1,1:-1,1:-1])/2.
			a_down = K_down/(dz**2); a_down[np.isnan(a_down)] = 0.
			s_down = a_down*h_extend[1:-1, 1:-1,:-2]; s_down[np.isnan(s_down)] = 0.
			b_down = -K_down/dz; b_down[np.isnan(b_down)] = 0.
			b_down[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]] = -K[field.bottomNode[0], field.bottomNode[1], field.bottomNode[2]]/dz

			a_i = Cw/dt+a_right+a_left+a_front+a_back+a_up+a_down
			h_next = (Cw/dt*h_before+s_right+s_left+s_front+s_back+s_up+s_down+S+b_up+b_down)/a_i
			field.h = (1.-lr)*field.h+lr*h_next
			field.h[field.h > 0] = 0.

	except:
		field.dead_flag = True

	if np.min(field.h) < -1e+100:
		field.dead_flag = True