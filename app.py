from pathlib import Path

import cv2
import numpy as np

class App:
    """
    check keypoint matches for an image pair
    
    # VIEW MODE 
    displays self.batch_size keypoints at a time
    press n to move to next batch 
    press e to edit current batch -> EDIT MODE
    
    # EDIT MODE (if e was pressed)
     display one keypoint at a time
        press n to move to next keypoint (return to view mode if last keypoint)
        click to modify keypoint coordinates of image2 (the one on the right) 
    """
    def __init__(self, img1_f, img2_f, mkpts_f):
        # load mkpts  
        self.mkpts_f = mkpts_f
        self.mkpts1, self.mkpts2 = np.load(mkpts_f) # (10,2)

		# load images
        img1_v0 = self.read_img_f(img1_f)
        img2_v0 = self.read_img_f(img2_f)

        # side by side image with no annotations drawn
        self.img_v0 = np.hstack((img1_v0, img2_v0))

        # img to display with annotations drawn
        self.img = self.img_v0.copy() 

        # image w, h
        self.h, self.w = img1_v0.shape[:2]

        # circle (for drawing annotations)
        self.r = 5
        self.t = 2
        self.colours = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255)]

        # num mkpts 
        self.num_mkpts = self.mkpts1.shape[0]

        # batch mkpts 
        self.batch_size = 5
        self.mkpts1_batches = [self.mkpts1[i:i+self.batch_size, :] for i in range(0, self.num_mkpts, self.batch_size)]
        self.mkpts2_batches = [self.mkpts2[i:i+self.batch_size, :] for i in range(0, self.num_mkpts, self.batch_size)]

        self.num_batches = len(self.mkpts1_batches)
        self.batch_index = 0 

        # editing 
        self.edit = False # true when editing (i.e. not viewing) 
        self.edit_index = 0 # index within current batch (for editing)
    
    def read_img_f(self, img_f):
        img = cv2.imread(img_f, cv2.IMREAD_GRAYSCALE)
        img = cv2.cvtColor(img ,cv2.COLOR_GRAY2RGB)
        return img 
    
    def reset_img(self):
        """reset display image to have no annotations drawn"""
        self.img = self.img_v0.copy()  
    
    def get_cur_batch_size(self):
        """in case batch_size does not divide evenly into num_mkpts"""
        return self.mkpts1_batches[self.batch_index].shape[0]

    def draw_mkpt(self, mkpt1, mkpt2, colour):
        """draw a keypoint match"""
        cv2.circle(self.img, (mkpt1[0], mkpt1[1]), self.r, colour, self.t)
        cv2.circle(self.img, (mkpt2[0] + self.w, mkpt2[1]), self.r, colour, self.t)
    
    def draw_mkpts(self):
        """draw a batch of keypoint matches"""
        mkpts1_batch = self.mkpts1_batches[self.batch_index]
        mkpts2_batch = self.mkpts2_batches[self.batch_index]

        for i in range(self.get_cur_batch_size()): 
            mkpt1 = mkpts1_batch[i]
            mkpt2 = mkpts2_batch[i]

            self.draw_mkpt(mkpt1, mkpt2, self.colours[i])
    
    def __call__(self):
        # mouse click event handler
        def draw_circle(event, x_click, y_click, flags, param):
            # do nothing if in view mode 
            if not self.edit: return 

            # mouse clicked in image 2 
            if event == cv2.EVENT_LBUTTONDOWN and x_click > self.w:
                # update keypoint coordinates in image 2 
                x2 = x_click - self.w
                y2 = y_click 
                self.mkpts2_batches[self.batch_index][self.edit_index] = np.array([x2, y2]) 

                mkpt1 = self.mkpts1_batches[self.batch_index][self.edit_index]
                mkpt2 = self.mkpts2_batches[self.batch_index][self.edit_index]

                # draw updated keypoint match 
                self.reset_img() 
                self.draw_mkpt(mkpt1, mkpt2, self.colours[self.edit_index])

                cv2.imshow("app", self.img)
        
        # APP 
        cv2.namedWindow("app")
        cv2.setMouseCallback("app", draw_circle)
        
        # draw first batch of keypoints 
        self.draw_mkpts()

        while self.batch_index < self.num_batches:
            cv2.imshow("app", self.img)

            k = cv2.waitKey(1) & 0xFF 
            # VIEW MODE (view batches of keypoints)
            if not self.edit:
                # go to next batch 
                if k == ord('n'):
                    self.batch_index += 1 
                    
                    if self.batch_index == self.num_batches: break 

                    self.reset_img()
                    self.draw_mkpts()
                # edit mode -> edit current batch 
                elif k == ord('e'):
                    self.edit = True 
                    self.edit_index = 0 

                    # show first keypoint in current batch 
                    self.reset_img() 

                    mkpt1 = self.mkpts1_batches[self.batch_index][self.edit_index]
                    mkpt2 = self.mkpts2_batches[self.batch_index][self.edit_index] 

                    self.draw_mkpt(mkpt1, mkpt2, self.colours[self.edit_index])

            # EDIT MODE (edit current batch of keypoints)
            else:
                # next keypoint in current batch 
                if k == ord('n'):
                    self.edit_index += 1 

                    # reached last keypoint in batch -> return to view mode 
                    if self.edit_index == self.get_cur_batch_size():
                        self.edit = False 

                        self.reset_img() 
                        self.draw_mkpts()
                    # next keypoint match in current batch  
                    else:
                        self.reset_img() 

                        mkpt1 = self.mkpts1_batches[self.batch_index][self.edit_index]
                        mkpt2 = self.mkpts2_batches[self.batch_index][self.edit_index]

                        self.draw_mkpt(mkpt1, mkpt2, self.colours[self.edit_index])

        # recombine batches 
        self.mkpts1 = np.concatenate(self.mkpts1_batches)
        self.mkpts2 = np.concatenate(self.mkpts2_batches)

        # save to disk 
        mkpts = np.stack((self.mkpts1, self.mkpts2))
        np.save(self.mkpts_f, mkpts)

        cv2.destroyAllWindows()
