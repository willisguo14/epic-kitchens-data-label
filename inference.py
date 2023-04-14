from pathlib import Path
import numpy as np 
from kornia.feature import LoFTR

from loftr import loftr_inference

#* FOLDER STRUCTURE: 
# EPIC-KITCHENS
# 	P01
# 		rgb_frames
# 			P01_03
# 				.jpg 

# EPIC-KITCHENS-ANNOTATIONS
# 	P01
# 		annotations
# 			P01_03
# 				.npy
# 			P01_03.txt 

if __name__ == "__main__":
    dataset_dir_path = "/Users/willisguo/EPIC-KITCHENS"
    annotations_dir_path = "/Users/willisguo/Desktop/data-label/EPIC-KITCHENS-ANNOTATIONS" 

    dataset_dir = Path(dataset_dir_path)
    annotations_dir = Path(annotations_dir_path)
    
    annotations_dir.mkdir(exist_ok=True)

    n = 5 # pairs are every n frames 

    matcher = LoFTR(pretrained='indoor_new')

    counter = 0 # print output 

    for e1 in dataset_dir.iterdir():
        if e1.is_dir():
            dataset_participant_dir = dataset_dir / e1.name # EPIC-KITCHENS/P01
            dataset_rgb_frames_dir = dataset_participant_dir / 'rgb_frames' # EPIC-KITCHENS/P01/rgb_frames

            annotations_participant_dir = annotations_dir / e1.name # EPIC-KITCHENS-ANNOTATIONS/PO1
            annotations_participant_dir.mkdir(exist_ok=True)

            for e2 in dataset_rgb_frames_dir.iterdir():
                if e2.is_dir():
                    dataset_video_dir = dataset_rgb_frames_dir / e2.name # EPIC-KITCHENS/P01/rgb_frames/P01_03
                    num_frames = len(list(dataset_video_dir.glob('*.jpg')))

                    annotations_video_dir = annotations_participant_dir / e2.name # EPIC-KITCHENS-ANNOTATIONS/PO1/P01_03
                    annotations_video_dir.mkdir(exist_ok=True)
                    
                    annotations_video_f = annotations_participant_dir / f'{e2.name}.txt' # EPIC-KITCHENS-ANNOTATIONS/PO1/P01_03.txt
                    with open(annotations_video_f, 'w') as f:
                        for i in range(1, num_frames+1, n):
                            if i+n <= num_frames:
                                # write to txt file
                                img1 = f'frame_{str(i):0>{10}}'
                                img2 = f'frame_{str(i+n):0>{10}}'
                                f.write(f'{img1}\t{img2}\n')
                                    
                                # loftr inference
                                fname1 = dataset_video_dir / f'{img1}.jpg' # EPIC-KITCHENS/P01/rgb_frames/P01_03/{img1}.jpg
                                fname2 = dataset_video_dir / f'{img2}.jpg'

                                mkpts0, mkpts1 = loftr_inference(matcher, str(fname1), str(fname2))
                                mkpts = np.stack((mkpts0, mkpts1))

                                mkpts_save_path = annotations_video_dir / f'{img1}-{img2}.npy' # EPIC-KITCHENS-ANNOTATIONS/PO1/P01_03/{img1}-{img2}.npy
                                np.save(mkpts_save_path, mkpts) 
                                
                                # print output 
                                counter += 1
                                print(counter)
                    

                    


            
            
    