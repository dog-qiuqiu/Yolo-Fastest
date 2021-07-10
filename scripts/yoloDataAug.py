import cv2, os
import numpy as np
import imgaug as ia
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

#加载数据
def loadData(imgPath, txtPath):
	outBoxes = [] 
	tmpBoxes = []
	
	img = cv2.imread(imgPath)
	h, w , c = img.shape
	outImg = img.reshape(1, h, w, c)
	
	with open(txtPath, "r") as f:
		for line in f.readlines():
			bbox = line.strip().split(" ")
			label = bbox[0]
			cx, cy, bw, bh = float(bbox[1]) * w, float(bbox[2]) * h, float(bbox[3]) * w, float(bbox[4]) * h
			x1, y1 = cx - 0.5 * bw, cy - 0.5 * bh
			x2, y2 = cx + 0.5 * bw, cy + 0.5 * bh
			tmpBoxes.append(ia.BoundingBox(x1, y1, x2, y2, label))
			
	outBoxes.append(tmpBoxes)
	return outImg, outBoxes
	

#数据增强策略
def dataAug(imgPath, txtPath):	
	images, bbs= loadData(imgPath, txtPath)
	
	seq = iaa.Sequential([
		iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.05*255)),
		iaa.Affine(translate_px={"x": (1, 5)}),
		iaa.MultiplyAndAddToBrightness(mul=(0.5, 1.5), add=(-30, 30)),
		iaa.Sometimes(0.25,iaa.imgcorruptlike.MotionBlur(severity=(1,2))),
		iaa.Resize({"height": (0.75, 1.25), "width": (0.75, 1.25)}),
		iaa.CropAndPad(percent=(-0.25, 0.25)),
		iaa.JpegCompression(compression=(0, 66))
	])

	image_aug, bbs_aug = seq(images=images, bounding_boxes=bbs)
	return image_aug[0], bbs_aug[0]

#检查坐标是否越界
def check(x1, y1, x2, y2, ih, iw):
	if x1 < 0.:
		x1 = 0.
	if y1 < 0.:
		y1 = 0.
	if x2 > iw:
		x2 = iw
	if y2 > ih:
		y2 = ih
	return x1, y1, x2, y2
		
	
#输出增强后的图像与label
def write(outName, index, outImg, outBoxes, outPath):
	T = 0
	ih, iw, _ = outImg.shape
	for i in outBoxes:

		label = i.label
		x1, y1, x2 ,y2 = i.x1, i.y1, i.x2, i.y2
		
		#越界判断
		if x2 < 0 or y2 < 0 or x1 > iw or y1 > ih or x1 > x2 or y1 > y2:  #图像外的检测框
			continue
		x1, y1, x2 ,y2 = check(x1, y1, x2, y2, ih, iw)
		
		bw, bh = x2 - x1, y2 -y1
		cx, cy = x1 + 0.5 * bw, y1 + 0.5 * bh
		
		bw, bh = bw/iw, bh/ih
		cx, cy = cx/iw, cy/ih
		
		if(cx < 0 or cx > 1 or cy < 0 or cy > 1 or bw < 0 or bw > 1 or bh < 0 or bh > 1):
			print(x1, y1, x2, y2)
			print(cx, cy, bw, bh)
			print("error label")
		else:
			with open("%s/au_%s_%d.txt"%(outPath, outName, index), "a+") as f:
				f.write(str(label)+" "+str(cx)+" "+str(cy)+" "+str(bw)+" "+str(bh)+"\n") 
			
			T = 1
			#debug 
			# x1, y1 = iw * (cx - 0.5 * bw), ih * (cy - 0.5 * bh)
			# x2, y2 = iw * (cx + 0.5 * bw), ih * (cy + 0.5 * bh)
			# cv2.rectangle(outImg, (int(x1), int(y1)), (int(x2), int(y2)), (255, 255, 0), 3)
	
	if T == 1:
		cv2.imwrite("%s/au_%s_%d.jpg"%(outPath, outName, index), outImg)
	

if __name__ == '__main__':
	multiple = 5       #数据扩充的倍数
	inpPath = "train"   #数据集的路径 yolo训练格式
	outPath = "output"  #增强后数据集输出的路径

	for i in range(multiple):
		fileList = os.listdir(inpPath)
		for f in fileList:
			if f [-1] == "t":
				name = f[:-4]
				print("%s.jpg"%name)
				outImg, outBoxes = dataAug("%s/%s.jpg"%(inpPath, name), "%s/%s.txt"%(inpPath,name))
				write(name, i, outImg, outBoxes, outPath)


