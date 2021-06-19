from jetcam.utils import bgr8_to_jpeg
import torch
import cv2
import numpy as np
import time
import json
import trt_pose.coco
import trt_pose.models
import torchvision.transforms as transforms
import PIL.Image
from trt_pose.draw_objects import DrawObjects
from trt_pose.parse_objects import ParseObjects
import glob
import pprint
from torch2trt import TRTModule
from utils_func import draw_points_image, get_person_valid_coords
from tracking.follow_keypoint import KeypointFollow
import os

def open_files(inputDir, ext):
    """
    Open all files with the given path and extension
    Returns
    -------
    list
        A list of the files in the folder with the specified file extension. 
    """
    
    files =  glob.glob(inputDir + '/*' + ext)
    return files

def trtInit():
    print("Init TRTModule")
    model_trt = TRTModule()
    print("loading optimized model state")
    model_trt.load_state_dict(torch.load('/home/jetson/jetson/dizzy/trt_pose/tasks/human_pose/weights/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'))
    print("Done init TRTModule")
    return model_trt


def model_init(num_parts, num_links):
    print("Init pose model")
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).cuda().eval()

    model.load_state_dict(torch.load('resnet18_baseline_att_224x224_A_epoch_249.pth'))
    print("Done init model")
    return model

device = torch.device('cuda')
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)
topology = trt_pose.coco.coco_category_to_topology(human_pose)
  
def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def main():        
    frame_rate = 15
    prev = 0

    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    # model = model_init(num_parts, num_links)
    kf = KeypointFollow()

    model = trtInit()
    try:
        while True:
            videos = open_files('vid','.mp4')

            for video in videos:

                print('Current video: {}'.format(video))
                cap         =   cv2.VideoCapture(video)

                while True:
                    stamp = time.time()
                    time_elapsed = stamp - prev
                    
                    if time_elapsed > 1./frame_rate:
                        ret, frame = cap.read()
                        # cv2.imshow('org', frame)
                        frame = cv2.resize(frame, (640,480))
                        t0 = time.time()

                        org = frame

                        prev = time.time()

                        if ret:
                            image = cv2.resize(frame, (224,224))

                            data = preprocess(image)
                            cmap, paf = model(data) # optimized model
                            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()
                            counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
                            draw_objects(image, counts, objects, peaks)

                            people = get_points(counts,objects,peaks)
                            print("\n\nfound {} people".format(len(people)))
                            x, y = get_person_valid_coords(people)
                            
                            x_scaled = np.interp(x, [1, 224], [1,640])
                            y_scaled = np.interp(y, [1, 224], [1,480])
                            # print(x_scaled, y_scaled)

                            # phy = np.pi + np.arctan2(-y_scaled, -x_scaled)
                            # print(phy)

                            print("Scaled x, y:", round(x_scaled),round(y_scaled))
                            # draw_points_image(org, round(x_scaled), round(y_scaled))
                            kf.follow_function(x_scaled, y_scaled)
                            # pprint.pprint(people)
                            # pose_frame = image[:, ::-1, :]
                            # pose_frame = bgr8_to_jpeg(image[:, ::-1, :])
                            # cv2.imshow('output', image)
                            t1 = time.time()

                            print("Execution time (ms):", (t1 - t0)*1000)
                            # os.system('cls' if os.name=='nt' else 'clear')
                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            print("Unable to read image")
                            break
            
                cap.release()
                cv2.destroyAllWindows() 

    except Exception as e:
        print(e)


def get_points(counts,objects,peaks):
    height = 224
    width = 224
    k = topology.shape[0]
    count = counts[0]
    people = []
    for human in range(count):
        obj = objects[0][human]
        person = {}
        for key in range(obj.shape[0]):
            value = int(obj[key])
            if value >=0:
                peak = peaks[0][key][value]
                x,y = (round(float(peak[1])*width),round(float(peak[0])*height))
            else:
                x,y = -1,-1
            person[human_pose['keypoints'][key]]=(x,y)
        people.append(person)
    return people


if __name__ == "__main__":
    main()
