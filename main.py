import logging
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

import job_service
from db import schemas
from db.database import SessionLocal
from dual_momentum import dual_momentum, humanize_portfolio


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


SessionDep = Annotated[SessionLocal, Depends(get_db_session)]

app = FastAPI()
app.mount("/static", StaticFiles(directory="static"), name="static")

logger = logging.getLogger('uvicorn.error')

templates = Jinja2Templates(directory="templates")


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html.jinja",
    )


@app.get("/jobs", response_class=HTMLResponse)
async def jobs(request: Request, session: SessionDep, page: int = 0):
    limit = 20
    offset = page * limit
    jobs = job_service.get_jobs_paginated(session, limit, offset)
    count = job_service.count_jobs(session)

    return templates.TemplateResponse(
        request=request,
        name="jobs.html.jinja",
        context={"jobs": jobs, "count": count, "page": page, "limit": limit, "offset": offset},
    )


@app.get("/jobs/{job_id}", response_class=HTMLResponse)
async def details(request: Request, session: SessionDep, job_id: int = 0):
    job = job_service.get_job(session, job_id)

    if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail="Job not found")

    portfolio_path, trades_path = job_results_paths(job.id)
    portfolio = pd.read_csv(portfolio_path)
    trades = pd.read_csv(trades_path)

    return templates.TemplateResponse(
        request=request,
        name="details.html.jinja",
        context={"job": job, "portfolio": portfolio, "trades": trades},
    )


class ModelData(BaseModel):
    start_date: str
    end_date: str
    # include_ytd: bool | None = None
    initial_investment: float
    tickers: str
    single_absolute_momentum: str | None = None
    safe_asset: str
    rebalance_period: int
    lookback_period: int
    switching_cost: float


@app.post("/")
async def model(data: Annotated[ModelData, Form()], session: SessionDep):
    start_date = datetime.strptime(data.start_date, "%Y-%m")
    end_date = datetime.strptime(data.end_date, "%Y-%m")

    job = schemas.JobCreate(
        start_year=start_date.year,
        start_month=start_date.month,
        end_year=end_date.year,
        end_month=end_date.month,
        tickers=data.tickers,
        safe_asset=data.safe_asset,
        initial_investment=data.initial_investment,
        rebalance_period=data.rebalance_period,
        lookback_period=data.lookback_period,
        switching_cost=data.switching_cost,
        single_absolute_momentum=data.single_absolute_momentum,
    )
    job = job_service.create_job(session, job)

    portfolio, trades = dual_momentum(job)
    portfolio = humanize_portfolio(portfolio)

    portfolio_path, trades_path = job_results_paths(job.id)
    with open(portfolio_path, 'w') as f:
        f.write(portfolio.to_csv())
    with open(trades_path, 'w') as f:
        f.write(trades.to_csv())

    return RedirectResponse('/', status_code=HTTPStatus.FOUND)


def job_results_paths(job_id: int) -> tuple[str, str]:
    portfolio_path = f'./static/results/{job_id}-portfolio.csv'
    trades_path = f'./static/results/{job_id}-trades.csv'
    return portfolio_path, trades_path