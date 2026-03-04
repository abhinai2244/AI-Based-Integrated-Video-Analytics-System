from fastapi import FastAPI, Depends
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy.orm import Session
from sqlalchemy import func
import threading
import copy
import cv2
import uvicorn
import time

from database.db import engine, get_db, Base
from database.models import VehicleEvent, PeopleEvent, ANPRLog
from video_pipeline import VideoPipeline

# Create DB tables
Base.metadata.create_all(bind=engine)

app = FastAPI(title="AI Video Analytics System")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

pipeline = None
pipeline_thread = None
latest_frame = None

def start_pipeline():
    global pipeline
    # Load input video (Assuming input.mp4 is available)
    pipeline = VideoPipeline(source="../input.mp4")
    pipeline.start()

@app.on_event("startup")
def startup_event():
    global pipeline_thread
    pipeline_thread = threading.Thread(target=start_pipeline, daemon=True)
    pipeline_thread.start()

@app.on_event("shutdown")
def shutdown_event():
    global pipeline
    if pipeline:
        pipeline.stop()

def video_stream():
    global pipeline
    while True:
        if pipeline and pipeline.current_frame is not None:
            # Safely copy frame to avoid threading issues
            frame = copy.deepcopy(pipeline.current_frame)
            ret, buffer = cv2.imencode('.jpg', frame)
            if not ret:
                continue
            frame_bytes = buffer.tobytes()
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
        else:
             time.sleep(0.1)

@app.get("/video_feed")
def get_video_feed():
    """Streams the processed video frames over HTTP."""
    return StreamingResponse(video_stream(), media_type="multipart/x-mixed-replace; boundary=frame")

@app.get("/stats")
def get_stats(db: Session = Depends(get_db)):
    """API Endpoint to fetch aggregated stats."""
    
    events_v = db.query(VehicleEvent.direction, func.count(VehicleEvent.id)).group_by(VehicleEvent.direction).all()
    vehicles = {k: v for k, v in events_v}
    
    events_p = db.query(PeopleEvent.direction, func.count(PeopleEvent.id)).group_by(PeopleEvent.direction).all()
    people = {k: v for k, v in events_p}
    
    events_g = db.query(PeopleEvent.gender, func.count(PeopleEvent.id)).group_by(PeopleEvent.gender).all()
    genders = {k: v for k, v in events_g}
    
    return {
        "vehicles": {
            "north": vehicles.get("North", 0),
            "south": vehicles.get("South", 0)
        },
        "people": {
            "entering": people.get("Enter", 0),
            "exiting": people.get("Exit", 0)
        },
        "gender": {
            "male": genders.get("Male", 0),
            "female": genders.get("Female", 0),
            "unknown": genders.get("Unknown", 0)
        }
    }

@app.get("/recent_plates")
def get_recent_plates(db: Session = Depends(get_db)):
    """API Endpoint for ANPR logs."""
    logs = db.query(ANPRLog).order_by(ANPRLog.timestamp.desc()).limit(10).all()
    return [{"plate": l.plate_text, "confidence": l.confidence, "time": l.timestamp} for l in logs]

@app.get("/frs_logs")
def get_frs_logs(db: Session = Depends(get_db)):
    """API Endpoint for FRS logs."""
    logs = db.query(FRSLog).order_by(FRSLog.timestamp.desc()).limit(10).all()
    return [{"name": l.recognized_name, "confidence": l.confidence, "time": l.timestamp} for l in logs]

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)
