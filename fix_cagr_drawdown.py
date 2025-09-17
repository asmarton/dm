import pandas as pd

import dual_momentum as dm
import industry_trends as it
import services.job_service as job_service
import utils
from db import models
from db.database import SessionLocal
from services import it_job_service
import numpy as np

db = SessionLocal()

print('Fixing DM jobs')
jobs = job_service.get_all_jobs(db)

for job in jobs:
    view_model = dm.load_results(job)
    cagr = round(view_model.cagr, 2)
    drawdown = view_model.drawdowns.iloc[0]['Dual Momentum Maximum Drawdown']
    drawdown = float(drawdown[:-1])

    start_date = view_model.portfolio['Date'].iloc[0]
    end_date = view_model.portfolio['Date'].iloc[-1]

    benchmark_returns, _ = dm.compute_monthly_returns_with_fallback(job.benchmark)
    benchmark_returns = benchmark_returns.loc[start_date:end_date]
    benchmark = job.initial_investment * (1 + benchmark_returns).cumprod().iloc[-1]
    benchmark = benchmark[job.benchmark]
    job.cagr_benchmark = round(((benchmark / job.initial_investment) ** (12 / len(benchmark_returns.index)) - 1) * 100, 2)

    benchmark_prices = utils.get_closing_prices(job.benchmark)
    benchmark_prices = benchmark_prices[start_date:end_date]
    benchmark_monthly_closes = benchmark_prices.groupby(pd.Grouper(freq='ME')).nth(-1)
    benchmark_running_max = np.maximum.accumulate(benchmark_monthly_closes)
    benchmark_drawdown_pct = (benchmark_monthly_closes - benchmark_running_max) / benchmark_running_max * 100
    job.drawdown_benchmark = abs(round(benchmark_drawdown_pct.min(), 2))

    job.trades = len(view_model.trades)

    try:
        db.query(models.Job).filter(models.Job.id == job.id).update({
            'cagr': cagr,
            'drawdown': abs(drawdown),
            'trades': job.trades,
        })
        db.commit()
    except Exception as e:
        print('ERROR: ', e)


print(f'Fixed {len(jobs)} DM jobs')
print('Fixing IT jobs')

jobs = it_job_service.get_all_jobs(db)

for job in jobs:
    view_model = it.load_results(job)
    drawdown = view_model.drawdowns.iloc[0]['Strategy Drawdown']
    drawdown_benchmark = view_model.drawdowns.iloc[0][f'{job.benchmark} Drawdown']

    drawdown = float(drawdown[:-1])
    drawdown_benchmark = float(drawdown_benchmark[:-1].replace('\ndtype: float64', '').replace(job.benchmark, '').strip())

    try:
        db.query(models.IndustryTrendsJob).filter(models.IndustryTrendsJob.id == job.id).update({
            'cagr': view_model.cagr,
            'drawdown': abs(drawdown),
            'cagr_benchmark': view_model.cagr_benchmark,
            'drawdown_benchmark': abs(drawdown_benchmark),
        })
        db.commit()
    except Exception as e:
        print('ERROR: ', e)


print(f'Fixed {len(jobs)} IT jobs')
db.close()