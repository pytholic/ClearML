import argparse

from clearml import Task

task = Task.init(project_name="best_practices", task_name="example_script")

parser = argparse.ArgumentParser()
parser.add_argument("-lr", default=0.001, help="Initial learning rate")
parser.add_argument("-epochs", default=10, help="Total number of epochs")
args = parser.parse_args()

parameters = {"change_me": "value", "dropout": 0.5}

task.connect(parameters)

config_file_yaml = task.connect_configuration(
    name="yaml file", configuration="./config.yaml"
)

print(args.lr, args.epochs, parameters)
