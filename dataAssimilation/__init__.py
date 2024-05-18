import numpy as np
import copy
import sys

"""
process : 個体に関するクラス
att :
	params -> <np:float:(D,)> パラメータリスト
	field -> <field> if None -> createFieldで要作成。
	etcModule -> <ETcModule> if None -> 必要ならばcreateETcModuleで作成。 植物寄与がなく、蒸発量が別途与えられるなら不要。
Note :
-- このクラスを直接用いることはできない。継承し、利用者自らカスタムする必要がある。
"""
class Individual:
	def __init__(self, params = None, field = None, etcModule = None):
		self.params = params
		self.field = field
		self.etcModule = etcModule

	"""
	process : クラスのコピー
	output : <Individual class>
	"""
	def copy(self):
		params = copy.deepcopy(self.params)
		field = self.field.copy()
		if self.etcModule is None:
			return type(self)(params, field, None)
		else:
			etcModule = self.etcModule.copy()
			return type(self)(params, field, etcModule)

	"""
	process : スカラー倍
	input : val -> <float>
	output : <Individual class>
	"""
	def mul(self, val):
		params = self.params*val; h = val*self.field.getH()
		new_individual = type(self)(params)
		new_individual.createField(h)
		new_individual.createETcModule()

		return new_individual

	"""
	process : Individual class どうしの和の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __add__(self, another):
		params = self.params + another.params
		h = self.field.getH() + another.field.getH()

		new_individual = type(self)(params)
		new_individual.createField(h)
		new_individual.createETcModule()
		
		return new_individual

	"""
	process : Individual class どうしの差の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __sub__(self, another):
		params = self.params - another.params
		h = self.field.getH() - another.field.getH()

		new_individual = type(self)(params)
		new_individual.createField(h)
		new_individual.createETcModule()
		
		return new_individual

	"""
	process : Individual class どうしの積の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __mul__(self, another):
		params = self.params * another.params
		h = self.field.getH() * another.field.getH()

		new_individual = type(self)(params)
		new_individual.createField(h)
		new_individual.createETcModule()

		return new_individual

	"""
	process : スカラーによる除算
	input : val -> <float>
	output : <Individual class>
	"""
	def truediv(self, val):
		params = self.params/val; h = self.field.getH()/val
		new_individual = type(self)(params)
		new_individual.createField(h)
		new_individual.createETcModule()

		return new_individual

	"""
	process : 尤度の計算
	input :
		y -> <ndarray> センサデータセット。(N, )なshape
		R -> <ndarray> 観測の分散共分散行列。(N, N)なshape
	output : <float> 尤度
	"""
	def calcLikelihood(self, y, R):
		if self.field.dead_flag or (self.checkConstraints() == False):
			return 0.
		else:
			try:
				h = self.observe(); R_inv = np.linalg.inv(R)
				return np.exp(-0.5*((y-h).reshape((1, -1))@R_inv)@(y-h).reshape((-1, 1)))[0,0]
			except:
				return 0.

	"""
	process : fieldクラスの作成
	input : h -> <np array> マトリックポテンシャル
	"""
	def createField(self, h):
		print("Error@piRichards.dataAssimilation.__init__.Individuals.createField")
		print("function <createField> should be overwritten.")
		sys.exit()

	"""
	process : ETcModuleクラスの作成
	"""
	def createETcModule(self):
		pass

	"""
	observe : データ観測
	Note : 引数は無し
	"""
	def observe(self):
		print("Error@piRichards.dataAssimilation.__init__.Individuals.observe")
		print("function <observe> should be overwritten.")
		sys.exit()

	"""
	process : 制約の確認
	output : <bool> True -> 制約を満たす。
	Note : 引数は無し
	"""
	def checkConstraints(self):
		return True



"""
process : 個体に関するクラス。ただし、hは四則演算に含まない。
att :
	params -> <ndarray> パラメータリスト。(D, )なshape
	field -> <field class> if None -> createFieldで要作成。
	etcModule -> <ETcModule> if None -> 必要ならばcreateETcModuleで作成。 植物寄与がなく、蒸発量が別途与えられるなら不要。
Note :
-- このクラスを直接用いることはできない。継承し、利用者自らカスタムする必要がある。
"""
class Individual_withoutH(Individual):
	"""
	process : スカラー倍
	input : val -> <float>
	output : <Individual class>
	"""
	def mul(self, val):
		params = self.params*val
		new_individual = type(self)(params)
		new_individual.createField(self.field.getH())
		new_individual.createETcModule()

		return new_individual

	"""
	process : Individual class どうしの和の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __add__(self, another):
		params = self.params + another.params

		new_individual = type(self)(params)
		new_individual.createField(self.field.getH())
		new_individual.createETcModule()
		
		return new_individual

	"""
	process : Individual class どうしの差の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __sub__(self, another):
		params = self.params - another.params

		new_individual = type(self)(params)
		new_individual.createField(self.field.getH())
		new_individual.createETcModule()
		
		return new_individual

	"""
	process : Individual class どうしの積の定義
	input : another -> <Individual class>
	output : <Individual class>
	"""
	def __mul__(self, another):
		params = self.params * another.params

		new_individual = type(self)(params)
		new_individual.createField(self.field.getH())
		new_individual.createETcModule()

		return new_individual

	"""
	process : スカラーによる除算
	input : val -> <float>
	output : <Individual class>
	"""
	def truediv(self, val):
		params = self.params/val
		new_individual = type(self)(params)
		new_individual.createField(self.field.getH())
		new_individual.createETcModule()

		return new_individual