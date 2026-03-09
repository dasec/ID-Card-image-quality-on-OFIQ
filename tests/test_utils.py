import unittest
from oidiq.utils import OIDIQConfig, QualityMetricConfig, QualityMetric, deep_copy_dict

class TestOIDIQConfig(unittest.TestCase):
    def test_initialization(self):
        cfg_data = {'param1': 10, 'param2': 'value'}
        config = OIDIQConfig(cfg_data, description="Test config")
        self.assertEqual(config['param1'], 10)
        self.assertEqual(config['param2'], 'value')
        self.assertEqual(config['description'], "Test config")
    
    def test_description_default(self):
        cfg_data = {'param1': 10}
        config = OIDIQConfig(cfg_data)
        self.assertEqual(config['description'], "")

    def test_description_override(self):
        cfg_data = {'param1': 10, 'description': 'Old description'}
        config = OIDIQConfig(cfg_data, description="New description")
        self.assertEqual(config['description'], "New description")

    def test_deep_copy(self):
        original = {'a': 1, 'b': {'c': 2}, "d": [3]}
        copied = deep_copy_dict(original)
        self.assertEqual(copied, original)
        copied['b']['c'] = 3
        self.assertEqual(original['b']['c'], 2)
        copied['d'][0] = 4
        self.assertEqual(original['d'][0], 3)

    def test_config_is_deep_copied(self):
        nested = {'a': 1, 'b': {'c': 2}, "d": [3]}
        config = OIDIQConfig(nested)
        config['b']['c'] = 3
        self.assertEqual(nested['b']['c'], 2)
        nested['d'][0] = 4
        self.assertEqual(config['d'][0], 3)

class TestQualityMetricConfig(unittest.TestCase):
    def test_initialization(self):
        cfg_data = {"some_param": 42}
        config = QualityMetricConfig(cfg_data, name="metric1", description="Metric description")
        self.assertEqual(config['name'], 'metric1')
        self.assertEqual(config['description'], "Metric description")
        self.assertEqual(config['some_param'], 42)


    def test_create_quality_metric(self):
        cfg_data = {'name': 'metric1', 'description': 'Metric description'}
        config = QualityMetricConfig(cfg_data)
        metric = config.create_quality_metric(raw_value=0.75, score=85)
        self.assertIsInstance(metric, QualityMetric)
        self.assertEqual(metric.name, 'metric1')
        self.assertEqual(metric.raw_value, 0.75)
        self.assertEqual(metric.score, 85)
        self.assertEqual(metric.description, 'Metric description')


    
    def test_score_clipping(self):
        cfg_data = {'name': 'metric1'}
        config = QualityMetricConfig(cfg_data)
        metric_low = config.create_quality_metric(raw_value=0.3, score=-10)
        metric_high = config.create_quality_metric(raw_value=0.9, score=150)
        self.assertEqual(metric_low.score, 0)
        self.assertEqual(metric_high.score, 100)