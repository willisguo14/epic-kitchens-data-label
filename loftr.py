import numpy as np 
import cv2
import torch
import kornia as K
from kornia.feature import LoFTR

def load_torch_image(fname):
  img = K.image_to_tensor(cv2.imread(fname), False).float() /255.
  img = K.color.bgr_to_rgb(img)
  return img

# TODO: use kornia or just pytorch 
def loftr_inference(matcher, fname0, fname1):
	img0 = load_torch_image(fname0)
	img1 = load_torch_image(fname1)

	img0 = K.color.rgb_to_grayscale(img0)
	img1 = K.color.rgb_to_grayscale(img1)

	input_dict = {"image0": img0,
				"image1": img1}

	with torch.inference_mode():
		correspondences = matcher(input_dict)

	mkpts0 = correspondences['keypoints0'].cpu().numpy()
	mkpts1 = correspondences['keypoints1'].cpu().numpy()
	mconf = correspondences['confidence'].cpu().numpy()

	# sort mkpts from highest to lowest confidence
	indices = mconf.argsort()[::-1]

	mconf = mconf[indices]
	mkpts0 = mkpts0[indices]
	mkpts1 = mkpts1[indices]

	# keep top 10 most confident mkpts 
	mconf = mconf[:10].astype(int)
	mkpts0 = mkpts0[:10].astype(int)
	mkpts1 = mkpts1[:10].astype(int)

	return mkpts0, mkpts1 