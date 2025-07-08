from sqlalchemy.orm import Session

from db import schemas, models


def get_jobs_paginated(db: Session, limit: int, offset: int, user: str | None = None) -> list[type[models.Job]]:
    query = db.query(models.Job)
    if user is not None:
        query = query.where(models.Job.user == user)
    return query.order_by(models.Job.created_at.desc()).offset(offset).limit(limit).all()


def get_job(db: Session, id: int) -> schemas.Job | None:
    return db.query(models.Job).get(id)


def count_jobs(db: Session, user: str | None = None) -> int:
    query = db.query(models.Job)
    if user is not None:
        query = query.where(models.Job.user == user)
    return query.count()


def create_job(db: Session, job: schemas.JobCreate):
    db_job = models.Job(**job.model_dump())
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job
