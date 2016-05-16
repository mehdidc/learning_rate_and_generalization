from invoke import task
from data import load_data
from train import train_and_validate
from lightjob.db import RUNNING, SUCCESS, AVAILABLE

import numpy as np

@task
def insert_jobs():
    from lightjob.cli import load_db
    db = load_db()
    nb = 0
    nb += insert_jobset(jobset1(), db, where='jobset1')
    print('{} jobs added'.format(nb))

def insert_jobset(jobs, db, **kw):
    nb = 0
    for j in jobs:
        nb += db.safe_add_job(j, **kw)
    return nb

def jobset1():
    jobs = []
    dataset = dict(
        name='mnist',
        training_subset=0.02, valid_subset=0.1, test_subset=0.1
    )
    hp = dict(
        learning_rate_decay=1,
        momentum=0.9,
        batchsize=128,
        nb_epochs=500,
        augment=False,
        augment_params={}
    )
    content = dict(
        model_name='ciresan_4',
        dataset=dataset,
        seed=42
    )
    # no decay, no momentum
    for lr in np.logspace(-6, 1, 100):
        hp_cur = hp.copy()
        hp_cur['learning_rate'] = lr
        content_cur = content.copy()
        content_cur['hp'] = hp_cur
        jobs.append(content_cur)
    return jobs

@task
def run_jobs(nb=1, where=None):
    from lightjob.cli import load_db
    db = load_db()
    kw = dict()
    if where is not None:
        kw['where'] = where
    jobs = db.jobs_with(state=AVAILABLE, **kw)
    jobs = jobs[0:int(nb)]
    print('Nb jobs to run : {}'.format(len(jobs)))
    for j in jobs:
        run_job_and_sync_db(j, db)

def run_job_and_sync_db(j, db):
    print(j['content'])
    db.modify_state_of(j['summary'], RUNNING)
    history = run_job(j['content'])
    db.job_update(j['summary'], {'history': history})
    db.modify_state_of(j["summary"], FINISHED)

def run_job(content):
    ct = content
    np.random.seed(ct['seed'])
    train, valid, test = load_data(**ct['dataset'])
    return train_and_validate(train, valid, test, hp=ct['hp'])

@task
def test():
    j = np.random.choice(jobset1())
    run_job(j)
