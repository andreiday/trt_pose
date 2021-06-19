import cv2

def draw_points_image(frame, x, y):
    frame = cv2.circle(frame, (x, y), radius=10, color=(0, 0, 255), thickness=-1)
    return frame

def get_person_valid_coords(people):
    for idx in range(len(people)):
        person = people[idx]
        # print("person id:", idx)
        # print("id: {}, person: {}".format(idx, person))
        for key in person:
            x, y = people[idx][key]
            if (x, y) != (-1, -1):
                print("Tracking: ", key, x, y)
            
                return x, y
