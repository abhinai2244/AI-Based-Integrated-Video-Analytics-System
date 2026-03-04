from sqlalchemy import Column, Integer, String, Float, DateTime
from datetime import datetime
from database.db import Base, engine

class VehicleEvent(Base):
    __tablename__ = "vehicle_events"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    vehicle_type = Column(String, index=True)
    direction = Column(String)  # "North", "South", "In", "Out"
    timestamp = Column(DateTime, default=datetime.utcnow)

class PeopleEvent(Base):
    __tablename__ = "people_events"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    direction = Column(String)  # "Enter", "Exit"
    gender = Column(String, default="Unknown")  # "Male", "Female", "Unknown"
    timestamp = Column(DateTime, default=datetime.utcnow)

class FRSLog(Base):
    __tablename__ = "frs_logs"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    recognized_name = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)

class ANPRLog(Base):
    __tablename__ = "anpr_logs"
    id = Column(Integer, primary_key=True, index=True)
    track_id = Column(Integer, index=True)
    plate_text = Column(String, index=True)
    confidence = Column(Float)
    timestamp = Column(DateTime, default=datetime.utcnow)


def init_db():
    Base.metadata.create_all(bind=engine)
