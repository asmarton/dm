from sqlalchemy.orm import Session

import utils
from db import schemas, models


def get_jobs_paginated(
    db: Session, limit: int, offset: int, user: str | None = None
) -> list[type[models.IndustryTrendsJob]]:
    query = db.query(models.IndustryTrendsJob)
    if user is not None:
        query = query.where(models.IndustryTrendsJob.user == user)
    return query.order_by(models.IndustryTrendsJob.created_at.desc()).offset(offset).limit(limit).all()


def get_job(db: Session, id: int) -> schemas.IndustryTrendsJob | None:
    job: models.IndustryTrendsJob = db.query(models.IndustryTrendsJob).get(id)
    if not job:
        return None
    job = schemas.IndustryTrendsJob.model_validate({**job.__dict__, 'tickers': utils.tickers_to_list(job.tickers)})
    return job


def count_jobs(db: Session, user: str | None = None) -> int:
    query = db.query(models.IndustryTrendsJob)
    if user is not None:
        query = query.where(models.IndustryTrendsJob.user == user)
    return query.count()


def create_job(db: Session, job: schemas.IndustryTrendsJobCreate):
    db_job = models.IndustryTrendsJob(**{**job.model_dump(), 'tickers': ','.join(job.tickers)})
    db.add(db_job)
    db.commit()
    db.refresh(db_job)
    return db_job
