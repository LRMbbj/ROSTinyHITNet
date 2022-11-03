import sys
from threading import Thread
import time
from turtle import update
from typing_extensions import Self
import cv2
import torch
from Dataloader import StreamLoader
from dataset.utils import np2torch
import numpy as np
import pytorch_lightning as pl
from models import build_model
from matplotlib import pyplot as plt
# import visdom
import rospy
from rospy import Header
from sensor_msgs.msg import Image


# vis = visdom.Visdom()


class PredictModel(pl.LightningModule):
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters()
        self.model = build_model(args)

    def forward(self, left, right):
        left = left * 2 - 1
        right = right * 2 - 1
        return self.model(left, right)


def fixdim(x):
    x = x[..., [2, 1, 0]]
    x = np.transpose(x, (0, 3, 1, 2))
    if x.dtype == np.uint8:
        x = x.astype(np.float32) / 255
    x = torch.from_numpy(x.copy())
    return x


class Predict():
    def __init__(self, dataloader, model):
        # Thread.__init__(self)
        self.dataloader = dataloader
        self.model = model
        self.xscale = 6
        self.yscale = 12
        self.pubs = []
        for k in range(6):
            self.pubs.append(rospy.Publisher('Stereo_Matcher/uav_{}/disp'.format(k), Image, queue_size=1))
        
    def StereoPub(self, imgs):
        height = imgs.shape[0]
        for k in range(6):
            # cv2.imshow('StereoMatcher/uav_{}/disp'.format(k), imgs[height // 6 * k:height // 6 * (k + 1) - 1, :])
            img_data = imgs[height // 6 * k:height // 6 * (k + 1) - 1, :]
            img_pack=Image()
            header = Header(stamp=rospy.Time.now())
            header.frame_id = 'map'
            img_pack.height = height // 6
            img_pack.width = imgs.shape[1]
            img_pack.encoding ='mono8'
            img_pack.data = img_data.tobytes()
            #print(imgdata)
            #image_temp.is_bigendian=True
            img_pack.header = header
            img_pack.step = imgs.shape[1]
            self.pubs[k].publish(img_pack)
        

    @torch.no_grad()
    def run(self) -> None:
        print("Going to play..")
        cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)
        

        for limgs, rimgs in self.dataloader:
            img = np.concatenate((limgs[0], rimgs[0]), axis=1)
            limg = limgs[0]
            rimg = rimgs[0]
            for k in range(1, 6):
                img = np.concatenate((img, np.concatenate((limgs[k], rimgs[k]), axis=1)), axis=0)
                limg = np.concatenate((limg, limgs[k]), axis=0)
                rimg = np.concatenate((rimg, rimgs[k]), axis=0)

            disps = None
            t0 = time.time()
            
            # leftlist = fixdim(limgs).cuda()
            # rightlist = fixdim(rimgs).cuda()
            
            limg = cv2.resize(limg, None, fx=1/self.xscale, fy=1/self.yscale, interpolation=cv2.INTER_AREA)
            rimg = cv2.resize(rimg, None, fx=1/self.xscale, fy=1/self.yscale, interpolation=cv2.INTER_AREA)
            
            for k in range(1):
            
                left = np2torch(limg, bgr=True).cuda().unsqueeze(0)
                right = np2torch(rimg, bgr=True).cuda().unsqueeze(0)
                # left = leftlist[k:k+3]
                # right = rightlist[k:k+3]
                
                
                # torch.cuda.synchronize()
                t1 = time.time()
                
                pred = self.model(left, right)
                
                
                # torch.cuda.synchronize()
                # print("[index:{}] eql fps:{}".format(k, 1./(time.time() - t1)))
                disp = pred["disp"]
                disp = torch.clip(disp / 192 * 255, 0, 255).long()
                # disp[disp == 255] = 0

                # vis.histogram(disp.flatten(), win="Hist", opts={"numbins": 25})
                
                disp = disp.cpu().numpy()[0][0]
                disp = np.sqrt(disp / 255) * 255
                disp = disp.astype(np.uint8)
                
                if disps is None:
                    disps = disp
                else:
                    disps = np.concatenate((disps, disp), axis=1)
                    
            print("[total] eql fps:{}".format(1./(time.time() - t0)))
            
            
            disps = cv2.resize(disps, None, fy=self.yscale, fx=self.xscale, interpolation=cv2.INTER_LINEAR)
            self.StereoPub(disps)
            disps = cv2.applyColorMap(disps, cv2.COLORMAP_TURBO)
            
            img = np.concatenate((img, disps), axis=1)
            
            cv2.imshow("Video", img)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            # time.sleep(1 / 30)  # 按原帧率播放

        print("Video End..")
        self.dataloader.Stop()
        cv2.destroyAllWindows()

    def ShowVideo(self):
        print("Going to play..")
        cv2.namedWindow("Video", cv2.WINDOW_KEEPRATIO)
        cv2.namedWindow("Disp", cv2.WINDOW_KEEPRATIO)

        for limgs, rimgs in self.dataloader:
            
            img = np.concatenate((limgs[0], rimgs[0]), axis=1)
            for k in range(1, 6):
                img = np.concatenate((img, np.concatenate((limgs[k], rimgs[k]), axis=1)), axis=0)
            cv2.imshow("Video", img)

            key = cv2.waitKey(1)
            if key == ord("q"):
                break

            time.sleep(1 / 30)  # 按原帧率播放

        print("Video End..")
        self.dataloader.Stop()
        cv2.destroyAllWindows()


class LoadVideo(Thread):
    def __init__(self, cam_path, dataloader):
        super().__init__()
        self.cam_path = cam_path
        self.dataloader = dataloader

    def run(self):
        print("stream predict running..")
        lcap = cv2.VideoCapture(self.cam_path["left"])
        rcap = cv2.VideoCapture(self.cam_path["right"])

        fps = lcap.get(cv2.CAP_PROP_FPS)  # 视频平均帧率
        print("fps: {}".format(fps))

        ff = 0

        while(lcap.isOpened() and rcap.isOpened() and self.dataloader.enable):
            lcap.grab()
            rcap.grab()

            success, im = lcap.retrieve()
            if success:
                limg = np.array(im)

            success, im = rcap.retrieve()
            if success:
                rimg = np.array(im)

            self.dataloader.update(limg, rimg)
            time.sleep(1/30)


class LoadVideoRos():
    def __init__(self, dataloader):
        super().__init__()
        self.dataloader = dataloader

    def callbackGenerator(self, index, tag):
        def callback(img):
            self.dataloader.updateSplit(np.frombuffer(img.data, dtype=np.uint8).reshape((img.height, img.width, 3)), index, tag)
        return callback


def GetTopic(index, cam_pos):
    return "/typhoon_h480_{}/stereo_camera/{}/image_raw".format(index, cam_pos)


class Args:
    def __init__(self) -> None: # HITNet_SF HITNetXL_SF StereoNet HITNet_KITTI
        self.model = "HITNetXL_SF"


if __name__ == "xx":
    cam_path = {"left": "./data/video/typhoon_h480_0_leftcam.avi",
                "right": "./data/video/typhoon_h480_0_rightcam.avi"}

    dataloader = StreamLoader()

    args = Args()

    model = PredictModel(args).eval()
    ckpt = torch.load("ckpt/hitnet_xl_sf_finalpass_from_tf.ckpt")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cuda()

    thread_load = LoadVideo(cam_path, dataloader)
    thread_pred = Predict(dataloader, model)

    thread_load.start()
    time.sleep(0.1)
    # thread_pred.start()
    thread_pred.run()

    print("All Thread Finished..")

if __name__ == "__main__":
    dataloader = StreamLoader()
    loader = LoadVideoRos(dataloader)
    
    args = Args()

    model = PredictModel(args).eval()
    
    # ckpt/hitnet_sf_finalpass.ckpt
    # ckpt/hitnet_xl_sf_finalpass_from_tf.ckpt
    # ckpt/stereo_net.ckpt
    
    ckpt = torch.load("ckpt/hitnet_xl_sf_finalpass_from_tf.ckpt")
    if "state_dict" in ckpt:
        model.load_state_dict(ckpt["state_dict"])
    else:
        model.model.load_state_dict(ckpt)
    model.cuda()

    rospy.init_node('Stereo_Matcher', anonymous=True)
    for i in range(6):
        for d in ("left", 'right'):
            rospy.Subscriber(
                GetTopic(i, d), Image, loader.callbackGenerator(i, d))
    time.sleep(1)
    pred = Predict(dataloader, model)
    pred.run()
