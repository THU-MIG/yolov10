from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

class YOLOv10(Model):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False):
        super().__init__(model=model, task=task, verbose=verbose)

    @property
    def task_map(self):
        """Map head to model, trainer, validator, and predictor classes."""
        return {
            "detect": {
                "model": YOLOv10DetectionModel,
                "trainer": YOLOv10DetectionTrainer,
                "validator": YOLOv10DetectionValidator,
                "predictor": YOLOv10DetectionPredictor,
            },
        }