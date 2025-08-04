import logging
import re
from datetime import datetime
from dateutil.relativedelta import relativedelta
from http import HTTPStatus
from typing import Annotated

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel

import utils
from services import job_service, it_job_service
import dual_momentum as dm
import industry_trends as it
from db import schemas
from db.database import SessionLocal


def get_db_session():
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


SessionDep = Annotated[SessionLocal, Depends(get_db_session)]

app = FastAPI()
app.mount('/static', StaticFiles(directory='static'), name='static')

logger = logging.getLogger('uvicorn.error')

templates = Jinja2Templates(directory='templates')


@app.get('/', response_class=HTMLResponse)
async def index(request: Request):
    user = request.cookies.get('user')
    return templates.TemplateResponse(
        request=request,
        name='index.html.jinja',
        context={'user': user},
    )


class JobFormData(BaseModel):
    user: str
    start_date: str
    end_date: str
    initial_investment: float
    tickers: str
    single_absolute_momentum: str | None = None
    safe_asset: str
    rebalance_period: int
    lookback_period: int
    switching_cost: float
    max_assets: int
    exclude_prev_month: bool = False
    benchmark: str | None = None


def job_form_data_to_schema(data: JobFormData) -> schemas.JobCreate:
    start_date = datetime.strptime(data.start_date, '%Y-%m')
    end_date = datetime.strptime(data.end_date, '%Y-%m')
    return schemas.JobCreate(
        start_year=start_date.year,
        start_month=start_date.month,
        end_year=end_date.year,
        end_month=end_date.month,
        tickers=utils.clear_tickers_list(data.tickers),
        safe_asset=data.safe_asset,
        initial_investment=data.initial_investment,
        rebalance_period=data.rebalance_period,
        lookback_period=data.lookback_period,
        switching_cost=data.switching_cost,
        single_absolute_momentum=data.single_absolute_momentum if data.single_absolute_momentum else None,
        max_assets=data.max_assets,
        exclude_prev_month=data.exclude_prev_month,
        user=data.user,
    )


@app.post('/')
async def model(data: Annotated[JobFormData, Form()], session: SessionDep):
    tickers_regex = r'^(\w+\s*,\s*)+(\w+)$'
    if not re.match(tickers_regex, data.tickers):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='Invalid tickers')

    job = job_form_data_to_schema(data)

    if data.max_assets > 1:
        results = dm.dual_momentum_multi(job)
    else:
        results = dm.dual_momentum(job)

    job.start_year = results.portfolio.index[0].year
    job.start_month = results.portfolio.index[0].month
    job = job_service.create_job(session, job)

    dm.save_results(job.id, results)

    response = RedirectResponse('/', status_code=HTTPStatus.FOUND)
    response.set_cookie(key='user', value=data.user, path='/', max_age=2592000)
    return response


@app.get('/jobs', response_class=HTMLResponse)
async def jobs(request: Request, session: SessionDep, page: int = 0, user_filter: str | None = None):
    limit = 10
    offset = page * limit
    user_filter = user_filter or None
    jobs = job_service.get_jobs_paginated(session, limit, offset, user_filter)
    count = job_service.count_jobs(session, user_filter)

    return templates.TemplateResponse(
        request=request,
        name='jobs.html.jinja',
        context={
            'jobs': jobs,
            'count': count,
            'page': page,
            'limit': limit,
            'offset': offset,
            'user_filter': user_filter,
        },
    )


@app.get('/jobs/{job_id}', response_class=HTMLResponse)
async def details(request: Request, session: SessionDep, job_id: int = 0):
    job = job_service.get_job(session, job_id)

    if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail='Job not found')

    view_model = dm.load_results(job)

    return templates.TemplateResponse(
        request=request,
        name='details.html.jinja',
        context={
            'job': view_model.job,
            'portfolio': view_model.portfolio,
            'trades': view_model.trades,
            'drawdowns': view_model.drawdowns,
            'ticker_info': view_model.ticker_info,
        },
    )


@app.post('/dm/compare-perf', response_class=HTMLResponse)
async def dm_compare_perf(request: Request, data: Annotated[JobFormData, Form()]):
    job = job_form_data_to_schema(data)
    perf, start_date, end_date = dm.compare_performance(job)
    benchmark_ticker = data.benchmark or 'SPY'
    benchmark, _ = dm.compute_monthly_returns_with_fallback(benchmark_ticker)
    benchmark = benchmark.loc[start_date:end_date]
    benchmark = data.initial_investment * (1 + benchmark).cumprod().iloc[-1]
    benchmark = benchmark[benchmark_ticker]
    return templates.TemplateResponse(
        request=request,
        name='compare-perf.html.jinja',
        context={
            'perf': perf,
            'benchmark': round(benchmark, 2),
            'benchmark_ticker': benchmark_ticker,
            'start_date': start_date,
            'end_date': end_date,
        },
    )


@app.post('/dm/allocate', response_class=HTMLResponse)
async def dm_allocate(request: Request, data: Annotated[JobFormData, Form()]):
    end_date = datetime.today().replace(day=1, hour=0, minute=0, second=0, microsecond=0)
    data.end_date = datetime.strftime(end_date, '%Y-%m')
    data.start_date = datetime.strftime(datetime.today() - relativedelta(months=data.lookback_period), '%Y-%m')
    job = job_form_data_to_schema(data)

    if data.max_assets > 1:
        results = dm.dual_momentum_multi(job)
        positions = results.last_selected_assets
    else:
        results = dm.dual_momentum(job)
        positions = [results.last_selected_asset]
    lookback_returns = (round(((1 + results.portfolio.filter(like='Return')).cumprod() - 1).iloc[-1:] * 100, 2)).astype(str) + '%'
    lookback_returns.drop('Dual Momentum Return', axis=1, inplace=True)

    return templates.TemplateResponse(
        request=request,
        name='allocation.html.jinja',
        context={
            'date': end_date,
            'positions': positions,
            'lookback_returns': lookback_returns,
            'lookback_period': job.lookback_period,
        }
    )


@app.get('/industry-trends', response_class=HTMLResponse)
async def industry_trends_index(request: Request):
    user = request.cookies.get('user')
    return templates.TemplateResponse(
        request=request,
        name='industry-trends/index.html.jinja',
        context={'user': user, 'etfs': it.etfs},
    )


@app.post('/industry-trends', response_class=HTMLResponse)
async def industry_trends_create(
    request: Request, session: SessionDep, payload: Annotated[schemas.IndustryTrendsJobBase, Form()]
):
    results = it.timing_etfs(payload)

    job = schemas.IndustryTrendsJobCreate(
        initial_balance=payload.initial_balance,
        start_date=payload.start_date,
        end_date=payload.end_date,
        tickers=payload.tickers,
        up_period=payload.up_period,
        down_period=payload.down_period,
        vol_window=payload.vol_window,
        max_leverage=payload.max_leverage,
        target_volatility=payload.target_volatility,
        trade_cost_per_share=payload.trade_cost_per_share,
        trade_cost_min=payload.trade_cost_min,
        rebalance_threshold=payload.rebalance_threshold,
        benchmark=payload.benchmark,
        user=payload.user,
    )
    job = it_job_service.create_job(session, job)

    it.save_results(job.id, results)

    response = RedirectResponse('/industry-trends', status_code=HTTPStatus.FOUND)
    response.set_cookie(key='user', value=payload.user, path='/', max_age=2592000)
    return response


@app.get('/industry-trends/jobs/{job_id}', response_class=HTMLResponse)
async def industry_trends_details(request: Request, session: SessionDep, job_id: int = 0):
    job = it_job_service.get_job(session, job_id)

    if not job:
        raise HTTPException(status_code=HTTPStatus.NOT_FOUND, detail='Job not found')

    view_model = it.load_results(job)

    return templates.TemplateResponse(
        request=request,
        name='industry-trends/details.html.jinja',
        context={
            'model': view_model,
        },
    )


@app.get('/industry-trends/jobs', response_class=HTMLResponse)
async def industry_trends_jobs(request: Request, session: SessionDep, page: int = 0, user_filter: str | None = None):
    limit = 10
    offset = page * limit
    user_filter = user_filter or None
    jobs = it_job_service.get_jobs_paginated(session, limit, offset, user_filter)
    count = it_job_service.count_jobs(session, user_filter)

    return templates.TemplateResponse(
        request=request,
        name='industry-trends/jobs.html.jinja',
        context={
            'jobs': jobs,
            'count': count,
            'page': page,
            'limit': limit,
            'offset': offset,
            'user_filter': user_filter,
        },
    )
