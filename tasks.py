from invoke import task
from data import load_mnist
from train import train_and_validate

@task
def insert_jobs():
    pass

@task
def test():
    train, valid, test = load_mnist(training_subset=0.1)
    train_and_validate(train, valid, test, hp=dict(nb_epochs=10))
