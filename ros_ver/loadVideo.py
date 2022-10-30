import Dataloader
import cv2


global dataloader
dataloader = Dataloader.StreamLoader()


def load_video(cam_path):
    print("stream predict running..")
    lcap = cv2.VideoCapture(cam_path["left"])
    rcap = cv2.VideoCapture(cam_path["right"])

    fps = lcap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
    print("fps: {}".format(fps))

    ff = 0

    while(lcap.isOpened() and rcap.isOpened()):
        dataloader.update(lcap, rcap)
        key = cv2.waitKey(int(1000/60))
        if key == ord("x"):
            break
        print("loading... frame : {}".format(ff))
        ff += 1


if __name__ == "__main__":
    cam_path = {"left": "./data/video/l_cam001.mkv",
                "right": "./data/video/r_cam001.mkv"}

    load_video(cam_path)
