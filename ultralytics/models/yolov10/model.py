from ultralytics.engine.model import Model
from ultralytics.nn.tasks import YOLOv10DetectionModel
from .val import YOLOv10DetectionValidator
from .predict import YOLOv10DetectionPredictor
from .train import YOLOv10DetectionTrainer

from huggingface_hub import PyTorchModelHubMixin
from .card import card_template_text

class YOLOv10(Model, PyTorchModelHubMixin, model_card_template=card_template_text):

    def __init__(self, model="yolov10n.pt", task=None, verbose=False, 
                 names=None):
        super().__init__(model=model, task=task, verbose=verbose)
        if names is not None:
            setattr(self.model, 'names', names)

    def push_to_hub(self, repo_name, **kwargs):
        config = kwargs.get('config', {})
        config['names'] = self.names
        config['model'] = self.model.yaml['yaml_file']
        config['task'] = self.task
        kwargs['config'] = config
        super().push_to_hub(repo_name, **kwargs)

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