import dual_momentum as dm
import services.job_service as job_service
from db import models
from db.database import SessionLocal

db = SessionLocal()

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
db.close()