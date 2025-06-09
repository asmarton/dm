import logging
from datetime import datetime
from decimal import Decimal
from http import HTTPStatus
from typing import Annotated

from fastapi import FastAPI, Request, Form, Depends
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import crud
from db import schemas
from db.database import SessionLocal
from dual_momentum import dual_momentum


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
async def index(request: Request, session: SessionDep):
    jobs = crud.get_jobs(session)

    return templates.TemplateResponse(
        request=request,
        name="index.html.jinja",
        context={"jobs": jobs},
    )


class ModelData(BaseModel):
    start_date: str
    end_date: str
    # include_ytd: bool | None = None
    initial_investment: float
    tickers: str
    # single_absolute_momentum: str | None = None
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
    )
    job = crud.create_job(session, job)

    portfolio, trades = dual_momentum(job)

    with open(f'./static/results/{job.id}-trades.csv', 'w') as f:
        f.write(trades.to_csv())
    with open(f'./static/results/{job.id}-portfolio.csv', 'w') as f:
        f.write(portfolio.to_csv())

    return RedirectResponse('/', status_code=HTTPStatus.FOUND)
