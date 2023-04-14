from pathlib import Path

from app import App 

#* FOLDER STRUCTURE: 
# EPIC-KITCHENS
# 	P01
# 		rgb_frames
# 			P01_03
# 				.jpg 

# EPIC-KITCHENS-ANNOTATIONS
# 	P01
		# P01_03
		# 	.npy
		# P01_03.txt 

if __name__ == "__main__":
    dataset_dir_path = "/Users/willisguo/EPIC-KITCHENS"
    annotations_dir_path = "/Users/willisguo/Desktop/data-label/EPIC-KITCHENS-ANNOTATIONS" 
    
    dataset_dir = Path(dataset_dir_path)
    annotations_dir = Path(annotations_dir_path)

    for e1 in annotations_dir.iterdir():
        if e1.is_dir(): 
            dataset_participant_dir = dataset_dir / e1.name # EPIC-KITCHENS/P01
            dataset_rgb_frames_dir = dataset_participant_dir / 'rgb_frames' # EPIC-KITCHENS/P01/rgb_frames

            annotations_participant_dir = annotations_dir / e1.name # EPIC-KITCHENS-ANNOTATIONS/PO1

            for e2 in annotations_participant_dir.iterdir():
                if e2.suffix == '.txt': # EPIC-KITCHENS-ANNOTATIONS/PO1/P01_03.txt 
                    with open(e2, 'r') as f:
                        lines = f.readlines() 

                    dataset_video_dir = dataset_rgb_frames_dir / e2.stem # EPIC-KITCHENS/P01/rgb_frames/P01_03
                    annotations_video_dir = annotations_participant_dir / e2.stem # EPIC-KITCHENS-ANNOTATIONS/PO1/P01_03

					# iterate through txt file 
                    index = 0 
                    while index < len(lines):
                        line = lines[index].strip() 
                        
						# line marked with 'X' -> image pair has already been checked 
                        if line[-1] == 'X': 
                            index += 1
                            continue 

                        img1, img2 = line.split('\t')

                        mkpts_f = annotations_video_dir / f'{img1}-{img2}.npy' 
                        img1_f = dataset_video_dir / f'{img1}.jpg'
                        img2_f = dataset_video_dir / f'{img2}.jpg'

                        if mkpts_f.exists(): #! TEMP
                            app = App(str(img1_f), str(img2_f), str(mkpts_f))
                            app() 
                        
                        index += 1 

                        # print output
                        print(f'{img1} - {img2}') 
                    
                    #! TODO: how to get user to exit and save progress 
                    # write to txt file to mark progress
                    # with open(e2, 'w') as f:

                    #     # i < index -> image pair has been checked
                    #     for i in range(index):
                    #         line = lines[i].strip() 
                            
                    #         if line[-1] != 'X':
                    #             line = f'{line} X\n'
                            
                    #         f.write(line) 
                        
                    #     # i > index -> image pair has not been checked 
                    #     for i in range(index, len(lines)):
                    #         line = lines[i]
                    #         f.write(line) 

                
