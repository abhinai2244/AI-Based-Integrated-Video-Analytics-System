import datetime
import numpy as np

class TrackableObject:
    def __init__(self, objectID, centroid):
        # store the object ID, then initialize a list of centroids
        # using the current centroid
        self.objectID = objectID
        self.centroids = [centroid]
        # initialize a boolean used to indicate if the object has
        # already been counted or not
        self.counted = False


class PeopleCounter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.totalUp = 0
        self.totalDown = 0
        
        self.trackableObjects = {}

    def process_person(self, box, id):
        # Calculate centroid
        startX, startY, endX, endY = box
        centroid = (int((startX + endX) / 2.0), int((startY + endY) / 2.0))
        
        to = self.trackableObjects.get(id, None)
        
        event_registered = None

        if to is None:
            to = TrackableObject(id, centroid)
        else:
            y = [c[1] for c in to.centroids]
            direction = centroid[1] - np.mean(y)
            to.centroids.append(centroid)

            if not to.counted:
                if direction < 0 and centroid[1] < self.height // 2:
                    self.totalUp += 1
                    to.counted = True
                    event_registered = "Exit"
                
                elif direction > 0 and centroid[1] > self.height // 2:
                    self.totalDown += 1
                    to.counted = True
                    event_registered = "Enter"

        self.trackableObjects[id] = to

        if event_registered:
            return {"event": event_registered, "track_id": id}
            
        return {"event": None}
