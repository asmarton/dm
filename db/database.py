from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from utils import ROOT_DIR

SQLALCHEMY_DATABASE_URL = f'sqlite:///{ROOT_DIR / 'database.db'}'

engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={'check_same_thread': False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
