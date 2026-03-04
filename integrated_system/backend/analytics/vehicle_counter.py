import logging

class VehicleCounter:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        
        self.up_count = 0
        self.down_count = 0
        self.car_count = 0
        self.truck_count = 0
        
        self.tracker1 = set()
        self.tracker2 = set()
        
        self.dir_data = {}

    def get_direction(self, id, y):
        if id not in self.dir_data:
            self.dir_data[id] = y
            return "Unknown"
        else:
            diff = self.dir_data[id] - y
            if diff < 0:
                return "South"
            else:
                return "North"

    def count_obj(self, box, id, cls):
        cx = int(box[0] + (box[2] - box[0]) / 2)
        cy = int(box[1] + (box[3] - box[1]) / 2)

        direct = self.get_direction(id, cy)

        if cy <= int(self.height // 2):
            return {"event": None}

        w, h = self.width, self.height
        event_registered = None

        if direct == "South":
            if cy > (h - 300):
                if id not in self.tracker1:
                    self.down_count += 1
                    self.tracker1.add(id)

                    if cls == 2:
                        self.car_count += 1
                    elif cls == 7:
                        self.truck_count += 1
                        
                    event_registered = "South"

        elif direct == "North":
            if cy < (h - 150):
                if id not in self.tracker2:
                    self.up_count += 1
                    self.tracker2.add(id)

                    if cls == 2:
                        self.car_count += 1
                    elif cls == 7:
                        self.truck_count += 1

                    event_registered = "North"

        if event_registered:
            return {
                "event": event_registered,
                "track_id": id,
                "cls": cls
            }
        return {"event": None}
