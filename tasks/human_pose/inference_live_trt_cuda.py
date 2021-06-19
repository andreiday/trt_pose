from jetcam.csi_camera import CSICamera
from jetcam.utils import bgr8_to_jpeg
import torch
from torch2trt import TRTModule
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
import pprint
from utils_func import draw_points_image, get_person_valid_coords
from tracking.follow_keypoint import KeypointFollow
import os

def trtInit():
    print("Init TRTModule")
    model_trt = TRTModule()
    print("loading optimized model state")
    model_trt.load_state_dict(torch.load('weights/resnet18_baseline_att_224x224_A_epoch_249_trt.pth'))
    print("Done init TRTModule")
    return model_trt

device = torch.device('cuda')
mean = torch.Tensor([0.485, 0.456, 0.406]).cuda()
std = torch.Tensor([0.229, 0.224, 0.225]).cuda()

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
  

def preprocess(image):
    global device
    device = torch.device('cuda')
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]


def main():        
    frame_rate = 15
    prev = 0
    fps_time = 0
    print("Init Camera")
    camera = CSICamera(width=640, height=480)


    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    kf = KeypointFollow()

    model_trt = trtInit()

    while True:

        stamp = time.time()
        time_elapsed = stamp - prev

        if time_elapsed > 1./frame_rate:
            prev = time.time()
            frame = camera.read()
            
            # preprocess image for inference
            image = cv2.resize(frame, (224,224))
            data = preprocess(image)

            # infere, make a prediction on the image
            cmap, paf = model_trt(data)
            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

            # parse objects (keypoints), counts (how many?), and heatmap peaks from the (confidence map) and (part affinity field)
            counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
            
            # draw keypoints and skeleton onto image
            draw_objects(image, counts, objects, peaks)

            # get detected person's skeleton and coords
            people = get_points(counts,objects,peaks)

            print("\n\nfound {} people".format(len(people)))
            
            # if there's at least one person
            if len(people):
                # get valid coords (!= -1, -1) of the first detected keypoint in the first detected person
                x, y = get_person_valid_coords(people)
                
                # rescale
                x_scaled = np.interp(x, [1, 224], [1,640])
                y_scaled = np.interp(y, [1, 224], [1,480])

                #print(x_scaled, y_scaled)
                #phy = np.pi + np.arctan2(-y_scaled, -x_scaled)
                #print(phy)

                # recale image
                image_rescaled = cv2.resize(image, (640,480))
                print("Scaled x, y:", round(x_scaled),round(y_scaled))

                # draw point to be followed and follow it
                frame_point = draw_points_image(image_rescaled, round(x_scaled), round(y_scaled))
                kf.follow_function(x_scaled, y_scaled)

                # follower delimiting lines
                cv2.line(frame_point, (0,240), (640, 240), (255,0,0), 3)
                cv2.line(frame_point, (320,0), (320, 480), (255,0,0), 3)

                # fps counter
                cv2.putText(frame_point, "FPS: %f" % (1.0 / (time.time() - fps_time)), (10, 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                cv2.namedWindow('frame_point_track', cv2.WINDOW_NORMAL)
                cv2.imshow('frame_point_track', frame_point)
            
            fps_time = time.time()
            
            #cv2.namedWindow('output', cv2.WINDOW_NORMAL)
            #cv2.imshow('output', image)

            # os.system('cls' if os.name=='nt' else 'clear')

        if cv2.waitKey(1) & 0xFF == ord('q'):            
            cv2.destroyAllWindows() 
            break

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
