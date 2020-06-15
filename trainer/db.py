from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker

from trainer.lib.config import config, DB_CON_KEY

engine = create_engine(config[DB_CON_KEY])
Session = sessionmaker(bind=engine)
Base = declarative_base()
