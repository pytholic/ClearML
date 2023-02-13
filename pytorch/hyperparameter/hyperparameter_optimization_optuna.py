import torch
import torch.nn as nn
import torch.nn.functional as F
from clearml import Task
from clearml.automation import (
    DiscreteParameterRange,
    HyperParameterOptimizer,
    UniformParameterRange,
)
from clearml.automation.optuna import OptimizerOptuna


# Define model classes
class Net1(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.5)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout2(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


class Net2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 32, 5, padding="same")
        self.conv2 = nn.Conv2d(32, 64, 5, padding="same")
        self.conv3 = nn.Conv2d(64, 128, 5, padding="same")
        self.dropout1 = nn.Dropout2d(0.25)
        self.dropout2 = nn.Dropout2d(0.25)
        self.dropout3 = nn.Dropout2d(0.5)

        x = torch.randn(3, 32, 32).view(-1, 3, 32, 32)
        self._to_linear = None
        self.convs(x)

        self.fc1 = nn.Linear(self._to_linear, 128)
        self.fc2 = nn.Linear(128, 10)

    def convs(self, x):
        x = F.max_pool2d(F.relu(self.conv1(x)), (2, 2))
        x = self.dropout1(x)
        x = F.max_pool2d(F.relu(self.conv2(x)), (2, 2))
        x = self.dropout2(x)
        x = F.max_pool2d(F.relu(self.conv3(x)), (2, 2))
        x = self.dropout3(x)

        if self._to_linear is None:
            self._to_linear = x[0].shape[0] * x[0].shape[1] * x[0].shape[2]
        return x

    def forward(self, x):
        x = self.convs(x)
        x = x.view(-1, self._to_linear)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


# Callback function
def job_complete_callback(
    job_id,  # type: str
    objective_value,  # type: float
    objective_iteration,  # type: int
    job_parameters,  # type: dict
    top_performance_job_id,  # type: str
):
    print(
        "Job completed!", job_id, objective_value, objective_iteration, job_parameters
    )

    if job_id == top_performance_job_id:
        print(f"Top performance experiment id: {top_performance_job_id}")


# Create a new task
task = Task.init(
    project_name="hyperparameter-optimization",
    task_name="Hyperparameter optimization - multiple iterations",
)

optimizer = HyperParameterOptimizer(
    base_task_id="0467f868e9154f3ab72727c141614d52",
    # set hyperparameters to be optimizaed
    hyper_parameters=[
        UniformParameterRange("Args/lr", min_value=0.01, max_value=0.3, step_size=0.05),
        DiscreteParameterRange("Args/net", values=["Net1", "Net2"]),
        DiscreteParameterRange("Args/epochs", values=[5, 10]),
    ],
    # setting the objective metric
    objective_metric_title="test",
    objective_metric_series="loss",
    objective_metric_sign="min",
    # set optimizer
    optimizer_class=OptimizerOptuna,
    # configure optimization parameters
    optimization_time_limit=10.0,  # minutes
    # compute_time_limit=2.0,
    total_max_jobs=10,
    max_number_of_concurrent_tasks=5,
    pool_period_min=0.2,
    min_iteration_per_job=1,
    max_iteration_per_job=5,
)

# This will automatically create and print the optimizer new task id
# for later use. if a Task was already created, it will use it.
optimizer.set_report_period(0.2)
optimizer.start_locally(job_complete_callback=job_complete_callback)

# wait until optimization completed or timed-out
optimizer.wait()

# top_exp = optimizer.get_top_experiments(top_k=3)
# print([t.id for t in top_exp])

top_exp = optimizer.get_top_experiments_details(top_k=3)
print(top_exp)

# make sure we stop all jobs
optimizer.stop()

print("Completed!")
