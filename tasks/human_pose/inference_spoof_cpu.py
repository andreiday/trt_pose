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
import math
from utils_func import draw_points_image, get_person_valid_coords
from tracking.follow_keypoint import KeypointFollow

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


def model_init(num_parts, num_links):
    print("Init pose model")
    model = trt_pose.models.resnet18_baseline_att(num_parts, 2 * num_links).eval()

    model.load_state_dict(torch.load('weights/resnet18_baseline_att_224x224_A_epoch_249.pth', map_location=torch.device('cpu')))
    print("Done init model")
    return model


device = torch.device('cpu')
mean = torch.Tensor([0.485, 0.456, 0.406])
std = torch.Tensor([0.229, 0.224, 0.225])

with open('human_pose.json', 'r') as f:
    human_pose = json.load(f)

topology = trt_pose.coco.coco_category_to_topology(human_pose)
joints = human_pose['keypoints']

def preprocess(image):
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = PIL.Image.fromarray(image)
    image = transforms.functional.to_tensor(image).to(device)
    image.sub_(mean[:, None, None]).div_(std[:, None, None])
    return image[None, ...]

def main():     
    # todo
    # implement threads for inference and camera processing
    # check if improvements
   
    frame_rate = 20
    prev = 0
    fps_time = 0
    num_parts = len(human_pose['keypoints'])
    num_links = len(human_pose['skeleton'])
    kf = KeypointFollow()

    parse_objects = ParseObjects(topology)
    draw_objects = DrawObjects(topology)
    model = model_init(num_parts, num_links)

    try:
        people = {}
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
                        frame = cv2.resize(frame, (640,480))
                        org = frame
                        # cv2.imshow('org', frame)

                        prev = time.time()

                        if ret:
                            image = cv2.resize(frame, (224,224))
                            
                            data = preprocess(image)
                            cmap, paf = model(data) # unoptimized model
                            cmap, paf = cmap.detach().cpu(), paf.detach().cpu()

                            counts, objects, peaks = parse_objects(cmap, paf)#, cmap_threshold=0.15, link_threshold=0.15)
                            
                            # draw keypoints on the frame
                            draw_objects(image, counts, objects, peaks)

                            # get people found in frame
                            # return a list of dicts with keypoint and coords
                            people = get_points(counts,objects,peaks)
                            print("found {} people".format(len(people)))
                            # person_angles = get_angles(people)

                            x, y = get_person_valid_coords(people)
                            x_scaled = np.interp(x, [1, 224], [1,640])
                            y_scaled = np.interp(y, [1, 224], [1,480])
                            # print(x_scaled, y_scaled)

                            # phy = np.pi + np.arctan2(-y_scaled, -x_scaled)
                            # print(phy)

                            print(round(x_scaled),round(y_scaled))
                            # draw_points_image(org, round(x_scaled), round(y_scaled))
                            print("Scaled x, y:", round(x_scaled),round(y_scaled))
                            # draw_points_image(org, round(x_scaled), round(y_scaled))
                            kf.follow_function(x_scaled, y_scaled)
                            # keypoints = get_keypoints(people)
                            # pprint.pprint(people)
                            # pprint.pprint(keypoints)
                            # pose_frame = image[:, ::-1, :]
                            # pose_frame = bgr8_to_jpeg(image[:, ::-1, :])
                            # fps counter
                            fps = (1.0 / (time.time() - fps_time))
                            print("FPS: %f" % fps)
                            # os.system('cls' if os.name=='nt' else 'clear')
                            fps_time = time.time()

                            # cv2.imshow('output', image)

                            if cv2.waitKey(1) & 0xFF == ord('q'):
                                break
                        else:
                            print("Unable to read image")
                cap.release()
                cv2.destroyAllWindows() 

    except Exception as e:
        print("Exception: ", e)


def get_points(counts, objects, peaks):
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
