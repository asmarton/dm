import dual_momentum as dm
import industry_trends as it
import services.job_service as job_service
from db import models
from db.database import SessionLocal
from services import it_job_service

db = SessionLocal()

print('Fixing DM jobs')
jobs = job_service.get_all_jobs(db)

for job in jobs:
    view_model = dm.load_results(job)
    cagr = round(view_model.cagr, 2)
    drawdown = view_model.drawdowns.iloc[0]['Dual Momentum Maximum Drawdown']
    drawdown = float(drawdown[:-1])
    db.query(models.Job).filter(models.Job.id == job.id).update({
        'cagr': cagr,
        'drawdown': abs(drawdown),
    })

db.commit()


print(f'Fixed {len(jobs)} DM jobs')
print('Fixing IT jobs')

jobs = it_job_service.get_all_jobs(db)

for job in jobs:
    view_model = it.load_results(job)
    drawdown = view_model.drawdowns.iloc[0]['Strategy Drawdown']
    drawdown_benchmark = view_model.drawdowns.iloc[0][f'{job.benchmark} Drawdown']

    drawdown = float(drawdown[:-1])
    drawdown_benchmark = float(drawdown_benchmark[:-1].replace('\ndtype: float64', '').replace(job.benchmark, '').strip())

    db.query(models.IndustryTrendsJob).filter(models.IndustryTrendsJob.id == job.id).update({
        'cagr': view_model.cagr,
        'drawdown': abs(drawdown),
        'cagr_benchmark': view_model.cagr_benchmark,
        'drawdown_benchmark': abs(drawdown_benchmark),
    })


db.commit()
print(f'Fixed {len(jobs)} IT jobs')
db.close()