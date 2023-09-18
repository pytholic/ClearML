"""
Folder example. Rest can be found at https://clear.ml/docs/latest/docs/guides/reporting/artifacts/
"""

from clearml import OutputModel, StorageManager, Task

VERSION = 3
task = Task.init(project_name="person-reid", task_name=f"model_v{VERSION}")
task.set_comment(
    "efficientnet-lite pretained model on msmt17 dataset \n showed best cross-eval results"
)
task.upload_artifact(
    "local folder",
    artifact_object="./person-reid/efficientnetlite/efficientnetlite-gnap",
)
