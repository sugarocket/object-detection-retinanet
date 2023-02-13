import time

import cv2
import numpy as np
from PIL import Image

from retinanet import Retinanet

if __name__ == "__main__":
    retinanet = Retinanet()
    #----------------------------------------------------------------------------------------------------------#
    #   mode Pattern for specifying tests：
    #   'predict'           Indicates a single image prediction or multiple
    #   'video'             Indicates video detection, you can call the camera or video for detection
    # 'dir_predict'         Indicates to traverse the folder to detect and save. Traverse the img folder by default
    #                       and save the 'img_out' folder
    #----------------------------------------------------------------------------------------------------------#
    mode = "video"
    #-------------------------------------------------------------------------#
    #   crop                Specifies whether to intercept the target after a single image is predicted
    #   count               Specifies whether to count targets
    #   crop、count only works when mode='predict'
    #-------------------------------------------------------------------------#
    crop            = False
    count           = False
    #----------------------------------------------------------------------------------------------------------#
    #   video_path          Used to specify the path of the video
    #
    #
    #   Only when mode='video'
    #----------------------------------------------------------------------------------------------------------#
    video_path      = 0
    video_save_path = ""
    video_fps       = 25.0

    test_interval   = 100

    dir_origin_path = "img/"
    dir_save_path   = "img_out/"

    if mode == "predict":
        '''
         1. If  you want to save the detected image, use r_image.save("img.jpg") to save it, and modify it directly in predict.py.
         2. If you want to get the coordinates of the prediction box, you can enter the retinanet.detect_image function and read the four values of top, left, bottom, and right in the drawing part.
         3. If you want to use the prediction frame to intercept the target, you can enter the retinanet.detect_image function, and use the obtained four values of top, left, bottom, and right in the drawing part
         Use the matrix method to intercept the original image.
         4. If you want to write extra words on the prediction map, such as the number of specific targets detected, you can enter the retinanet.detect_image function and judge the predicted_class in the drawing part,
         For example, judging if predicted_class == 'car': can judge whether the current target is a car, and then record the number. Use draw.text to write.
        '''
        while True:
            img = input('Input image filename:')
            try:
                image = Image.open(img)
            except:
                print('Open Error! Try again!')
                end = input('You wanna try again or end?')
                if end == 'end':
                    break
                continue
            else:
                r_image = retinanet.detect_image(image)
                r_image.show()

    elif mode == "dir_predict":
        import os

        from tqdm import tqdm

        img_names = os.listdir(dir_origin_path)
        for img_name in tqdm(img_names):
            if img_name.lower().endswith(('.bmp', '.dib', '.png', '.jpg', '.jpeg', '.pbm', '.pgm', '.ppm', '.tif', '.tiff')):
                image_path  = os.path.join(dir_origin_path, img_name)
                image       = Image.open(image_path)
                r_image     = retinanet.detect_image(image)
                if not os.path.exists(dir_save_path):
                    os.makedirs(dir_save_path)
                r_image.save(os.path.join(dir_save_path, img_name.replace(".jpg", ".png")), quality=95, subsampling=0)



    elif mode == "video":
        capture = cv2.VideoCapture(video_path)
        if video_save_path!="":
            fourcc  = cv2.VideoWriter_fourcc(*'XVID')
            size    = (int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)), int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)))
            out     = cv2.VideoWriter(video_save_path, fourcc, video_fps, size)

        ref, frame = capture.read()
        if not ref:
            raise ValueError("Cannot find camera")

        fps = 0.0
        while(True):
            t1 = time.time()

            ref, frame = capture.read()
            if not ref:
                break

            frame = cv2.cvtColor(frame,cv2.COLOR_BGR2RGB)

            frame = Image.fromarray(np.uint8(frame))

            frame = np.array(retinanet.detect_image(frame))

            frame = cv2.cvtColor(frame,cv2.COLOR_RGB2BGR)

            fps  = ( fps + (1./(time.time()-t1)) ) / 2
            print("fps= %.2f"%(fps))
            frame = cv2.putText(frame, "fps= %.2f"%(fps), (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("video",frame)
            c= cv2.waitKey(1) & 0xff
            if video_save_path!="":
                out.write(frame)

            if c==27:
                capture.release()
                break

        print("Video Detection Done!")
        capture.release()
        if video_save_path!="":
            print("Save processed video to the path :" + video_save_path)
            out.release()
        cv2.destroyAllWindows()

    else:
        raise AssertionError("Please specify the correct mode: 'predict', 'video' ")
