from sqlalchemy.orm import Session

from db import schemas
from db import models


def get_jobs(db: Session):
    return db.query(models.Job).all()


def create_job(db: Session, job: schemas.JobCreate):
    db_job = models.Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job
