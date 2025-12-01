import sys
import os
# Add parent directory to path to allow imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from experiment.evaluation_metrics import EvaluationMetrics
from model import GPT

evaluation_metrics = EvaluationMetrics(Model=GPT)
evaluation_metrics.measure_flop_count()
