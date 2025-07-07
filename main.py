import json
import logging
import pathlib
import re
from datetime import datetime
from http import HTTPStatus
from typing import Annotated

import fastapi.encoders
from fastapi import FastAPI, Request, Form, Depends, HTTPException, Query
from fastapi.responses import HTMLResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
import pandas as pd

import job_service
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


@app.post('/')
async def model(data: Annotated[JobFormData, Form()], session: SessionDep):
    start_date = datetime.strptime(data.start_date, '%Y-%m')
    end_date = datetime.strptime(data.end_date, '%Y-%m')

    tickers_regex = r'^(\w+\s*,\s*)+(\w+)$'
    if not re.match(tickers_regex, data.tickers):
        raise HTTPException(status_code=HTTPStatus.BAD_REQUEST, detail='Invalid tickers')

    job = schemas.JobCreate(
        start_year=start_date.year,
        start_month=start_date.month,
        end_year=end_date.year,
        end_month=end_date.month,
        tickers=dm.clear_tickers_list(data.tickers),
        safe_asset=data.safe_asset,
        initial_investment=data.initial_investment,
        rebalance_period=data.rebalance_period,
        lookback_period=data.lookback_period,
        switching_cost=data.switching_cost,
        single_absolute_momentum=data.single_absolute_momentum if data.single_absolute_momentum else None,
        user=data.user,
    )

    portfolio, trades, ticker_info = dm.dual_momentum(job)
    portfolio = dm.humanize_portfolio(portfolio)

    job.start_year = portfolio.index[0].year
    job.start_month = portfolio.index[0].month
    job = job_service.create_job(session, job)

    portfolio_path, trades_path, ticker_info_path = job_results_paths(job.id)
    with open(portfolio_path, 'w') as f:
        f.write(portfolio.to_csv())
    with open(trades_path, 'w') as f:
        f.write(trades.to_csv())
    with open(ticker_info_path, 'w') as f:
        f.write(json.dumps(ticker_info, default=fastapi.encoders.jsonable_encoder))

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

    portfolio_path, trades_path, ticker_info_path = job_results_paths(job.id)
    portfolio = pd.read_csv(portfolio_path)
    trades = pd.read_csv(trades_path)
    if pathlib.Path(ticker_info_path).exists():
        with open(ticker_info_path) as f:
            ticker_info_json = json.load(f)
        ticker_info = [dm.TickerInfo(**ticker_info) for ticker_info in ticker_info_json]
    else:
        ticker_info = []

    portfolio.index = portfolio.index + 1
    trades.index = trades.index + 1

    drawdowns = pd.DataFrame()
    for col in portfolio.columns:
        col_search = re.search(r'(.*) Balance', col)
        if col_search:
            asset = col_search.group(1)
            balance_col = f'{asset} Balance'
            running_max = portfolio[balance_col].cummax()
            drawdown_pct = (portfolio[balance_col] - running_max) / running_max * 100
            mdd = drawdown_pct.min()
            drawdowns[f'{asset} Maximum Drawdown'] = [f'{round(mdd, 2)}%']

    return templates.TemplateResponse(
        request=request,
        name='details.html.jinja',
        context={
            'job': job,
            'portfolio': portfolio,
            'trades': trades,
            'drawdowns': drawdowns,
            'ticker_info': ticker_info,
        },
    )


def job_results_paths(job_id: int) -> tuple[str, str, str]:
    portfolio_path = f'./static/results/{job_id}-portfolio.csv'
    trades_path = f'./static/results/{job_id}-trades.csv'
    ticker_info_path = f'./static/results/{job_id}-ticker-info.json'
    return portfolio_path, trades_path, ticker_info_path


@app.get('/industry-trends', response_class=HTMLResponse)
async def trend_following(request: Request):
    user = request.cookies.get('user')
    return templates.TemplateResponse(
        request=request,
        name='industry-trends.html.jinja',
        context={'user': user, 'etfs': it.etfs},
    )


@app.get('/industry-trends/results', response_class=HTMLResponse)
async def trend_following(request: Request, query: Annotated[it.TrendFollowingJob, Query()]):
    user = request.cookies.get('user')
    results = it.trend_following_strategy(query)

    balance = pd.concat([results.equity, results.holdings, results.cash, results.borrowed], axis=1).rename(columns={0: 'Equity', 1: 'Holdings', 2: 'Cash', 3: 'Borrowed'})

    monthly_returns = pd.DataFrame(results.equity.groupby(pd.Grouper(freq='ME')).nth(-1).pct_change() * 100).rename(columns={0: 'Return'})
    monthly_returns.index.name = 'Date'
    monthly_returns['Year'] = monthly_returns.index.year
    monthly_returns['Month'] = monthly_returns.index.month
    monthly_returns = monthly_returns.pivot(index='Year', columns='Month', values='Return')
    monthly_returns.columns = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
                        'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']
    monthly_returns['Yearly'] = ((1 + monthly_returns / 100).prod(axis=1) - 1) * 100
    monthly_returns.fillna('', inplace=True)

    trades = results.shares.diff().fillna(0)
    trades = trades.loc[~(trades==0).all(axis=1)]
    trades_count = trades.astype(bool).sum().sum()

    return templates.TemplateResponse(
        request=request,
        name='industry-trends.html.jinja',
        context={'user': user, 'etfs': it.etfs, 'query': query, 'results': results, 'balance': balance, 'monthly_returns': monthly_returns, 'trades': trades, 'trades_count': trades_count},
    )