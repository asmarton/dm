from sqlalchemy.orm import Session

from db import schemas, models


def get_jobs_paginated(db: Session, limit: int, offset: int, user: str | None = None, sort_cagr: str | None = None, sort_drawdown: str | None = None) -> list[type[models.Job]]:
    query = db.query(models.Job)
    if user is not None:
        query = query.where(models.Job.user == user)
    if sort_cagr is None and sort_drawdown is None:
        query = query.order_by(models.Job.created_at.desc())
    else:
        if sort_cagr == 'asc':
            query = query.order_by(models.Job.cagr.asc())
        elif sort_cagr == 'desc':
            query = query.order_by(models.Job.cagr.desc())
        if sort_drawdown == 'asc':
            query = query.order_by(models.Job.drawdown.asc())
        elif sort_drawdown == 'desc':
            query = query.order_by(models.Job.drawdown.desc())
    return query.offset(offset).limit(limit).all()


def get_all_jobs(db: Session) -> list[type[models.Job]]:
    return db.query(models.Job).all()


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
