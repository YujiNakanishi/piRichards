import numpy as np
import os
import sys
import struct

"""
process : STLファイルから計算領域に関するデータを作成
input :
	size -> <tuple of float> (x, y, z)の計算格子サイズ (m)
	filename -> <str> STLファイル名
	scale -> <str> STLファイルで採用されている長さ単位
	delimiter -> <str> STLファイルで採用されている区切り文字
	filetype -> <str: "ascii" or "binary"> ファイルタイプ
output :
	voxel -> <ndarray:bool> ボクセル情報。(Nx, Ny, Nz)なshape。
		if voxel[i, j, k] == True -> オブジェクト内側 (流体領域)
	top_cell -> <list of list> topCellの出力
	bottom_cell -> <list of list> bottomCellの出力
"""
def createCell(size, filename, scale = "mm", delimiter = " ", filetype = "binary"):
	getometry = STL(filename, scale, delimiter, filetype) #<STL>
	x_ran, y_ran, z_ran = getometry.getSize() #<tuple> 各軸のmin-max
	if scale == "mm":
		x_ran = (x_ran[0]/1000., x_ran[1]/1000.); y_ran = (y_ran[0]/1000., y_ran[1]/1000.); z_ran = (z_ran[0]/1000., z_ran[1]/1000.)

	#####ボクセルshape計算。ボクセル作成。
	shape = (int((x_ran[1]-x_ran[0])/size[0]), int((y_ran[1]-y_ran[0])/size[1]), int((z_ran[1]-z_ran[0])/size[2]))
	voxel = np.zeros(shape).astype(bool)

	#####各ボクセルに対して流体領域かどうかを確認
	for ix in range(shape[0]):
		x = (ix+0.5)*size[0]+x_ran[0]
		for iy in range(shape[1]):
			y = (iy+0.5)*size[1]+y_ran[0]
			for iz in range(shape[2]):
				z = (iz+0.5)*size[2]+z_ran[0]
				voxel[ix, iy, iz] = getometry.isIn(np.array([x, y, z]))
				
	return voxel, topCell(voxel), bottomCell(voxel)


"""
process : 地表面にある計算格子のインデックスを所得
input : 
	voxel -> <ndarray>
output -> <list of list> (x, y, z)のインデックス番号のリスト
"""
def topCell(voxel):
	topListX = []; topListY = []; topListZ = []

	for i in range(voxel.shape[0]):
		for j in range(voxel.shape[1]):
			for k, v in enumerate(reversed(voxel[i,j,:])):
				if v:
					topListX.append(i); topListY.append(j); topListZ.append(-k-1)
					break
	return [topListX, topListY, topListZ]

"""
process : 自由排水面にある計算格子のインデックスを所得
input : 
	voxel -> <ndarray>
output -> <list of list> (x, y, z)のインデックス番号のリスト
"""
def bottomCell(voxel):
	I, J, K = voxel.shape
	bottomListX = []; bottomListY = []; bottomListZ = []

	for i in range(voxel.shape[0]):
		for j in range(voxel.shape[1]):
			for k, v in enumerate(voxel[i,j,:]):
				if v:
					bottomListX.append(i); bottomListY.append(j); bottomListZ.append(k)
					break
	return [bottomListX, bottomListY, bottomListZ]

"""
class : STLファイルデータを格納したクラス
att :
	scale -> <str> stlで採用されている長さ単位
	patches -> <list of dictionary> stlファイルに書かれているPatchのリスト
"""
class STL:
	"""
	input : 
	filename -> <str> ファイル名
	scale -> <str> stlで採用されている長さ単位
	delimiter -> <str> stlで採用されている区切り文字。filetypeがasciiのときのみ寄与
	filetype -> <str: "ascii" or "binary"> ファイルタイプ
	"""
	def __init__(self, filename, scale = "mm", delimiter = " ", filetype = "binary"):
		self.patches = []; self.scale = scale

		##########ファイル読み込み
		if filetype == "ascii":
			#####ascii形式の場合
			with open(filename, "r") as file:
				for f in file:

					#####行を文字で区切る
					f = f[:-1].split(self.delimiter); f = [s for s in f if s != ""]
					
					#####新しくPatchを追加
					if "solid" in f:
						patch = {"name" : f[-1], "facet_normal" : None, "vertex" : None}
						self.patches.append(patch)
					#####facet_normal情報追加
					elif f[0] == "facet":
						x, y, z = float(f[2]), float(f[3]), float(f[4]) #<float> ベクトル値
						self.patches[-1]["facet_normal"] = np.array([[x, y, z]]) if (self.patches[-1]["facet_normal"] is None) else np.concatenate((self.patches[-1]["facet_normal"], np.array([[x, y, z]])), axis = 0)
					#####vertex情報追加
					elif f[0] == "vertex":
						x, y, z = float(f[1]), float(f[2]), float(f[3])
						self.patches[-1]["vertex"] = np.array([[x, y, z]]) if (self.patches[-1]["vertex"] is None) else np.concatenate((self.patches[-1]["vertex"], np.array([[x, y, z]])), axis = 0)
					else: pass #EOF等


				#####vertexのshapeを(Nt, 3, 3)に修正
				for p in self.patches:
					p["vertex"] = np.expand_dims(p["vertex"], axis = 1).reshape((-1, 3, 3))

		else:
			#####binary形式の場合
			filesize = os.path.getsize(filename) #<int> ファイルサイズ (or 未読のファイルデータ数)
			with open(filename, "rb") as file:

				#####未読ファイルデータ数がゼロになるまで(=EOFまで)
				while filesize > 0:
					#####Patchを追加
					patch = {"name" : file.read(80).decode(), "facet_normal" : None, "vertex" : None}
					self.patches.append(patch); filesize -= 80
					tri_num = struct.unpack("I", file.read(4))[0]; filesize -= 4 #<int> Patchの三角形の数

					for itri in range(tri_num):
						#####facet_normal情報読み込み
						xyz = [struct.unpack("f", file.read(4))[0] for i in range(3)]; filesize -= 12
						self.patches[-1]["facet_normal"] = np.array([xyz]) if (self.patches[-1]["facet_normal"] is None) else np.concatenate((self.patches[-1]["facet_normal"], np.array([xyz])), axis = 0)

						#####vertex情報読み込み
						tri1 = np.array([struct.unpack("f", file.read(4))[0] for i in range(3)])
						tri2 = np.array([struct.unpack("f", file.read(4))[0] for i in range(3)])
						tri3 = np.array([struct.unpack("f", file.read(4))[0] for i in range(3)]); filesize -= 36
						tri = np.stack((tri1, tri2, tri3), axis = 0); tri = np.expand_dims(tri, axis = 0)
						self.patches[-1]["vertex"] = tri if (self.patches[-1]["vertex"] is None) else np.concatenate((self.patches[-1]["vertex"], tri), axis = 0)

						unknown = file.read(2); filesize -= 2

	"""
	process : 領域サイズを抽出
	output : <list of tuple>
	"""
	def getSize(self):
		points = np.concatenate([patch["vertex"].reshape((-1, 3)) for patch in self.patches], axis = 0)
		x = points[:,0]; y = points[:,1]; z = points[:,2]

		return [(np.min(x), np.max(x)), (np.min(y), np.max(y)), (np.min(z), np.max(z))]

	"""
	process : Xがオブジェクト内側にあるかどうか判定
	input : 
		X -> <ndarray> 参照点
	output : <bool>
	"""
	def isIn(self, X):
		triangles = np.concatenate([patch["vertex"] for patch in self.patches], axis = 0) - X
		norm = np.sqrt(np.sum(triangles ** 2, axis = 2))
		
		winding_number = 0.
		for (A, B, C), (a, b, c) in zip(triangles, norm):
			winding_number += np.arctan2(np.linalg.det(np.array([A, B, C])), (a * b * c) + c * np.dot(A, B) + a * np.dot(B, C) + b * np.dot(C, A))

		return winding_number >= (2.*np.pi - 1e-10)