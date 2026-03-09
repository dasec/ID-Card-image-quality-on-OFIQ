import unittest
from oidiq.session import OIDIQPreprocessor, OIDIQMetricCreator, OIDIQSession
from oidiq.utils import QualityMetric, OIDIQConfig, QualityMetricConfig, creates, config


class DummyPreprocessor(OIDIQPreprocessor):
    @config("config2")
    def init_config(self, config):
        config["initialized"] = True

    @creates("output1", "output2")
    def foo(self, session, config1, config2):
        return config1["a"], config2["b"]

    @creates("output3")
    @config("config1", "config2")
    def bar(self, session, config1, config2):
        return config1["c"] + sum(config2["d"])

    @creates("output4")
    @config("config2")
    def baz(self, session, config2):
        return config2["initialized"]
    
    @creates("output5")
    @config("config1")
    def qux(self, session, config1):
        return config1["c"]
    
    @creates("output6")
    @config()
    def quux(self, session):
        return session
    
class SimpleDummyPreprocessor(OIDIQPreprocessor):
    @creates("simple_output")
    def simple_method(self, session, config):
        return config.get("a", None)
    
class MultiConfigPreprocessor(OIDIQPreprocessor):
    @creates("out1")
    def create_out1(self, session, config):
        return config["x"]
    @creates("out2")
    def create_out2(self, session, config):
        return config["x"] * 2


class TestOIDIQPreprocessor(unittest.TestCase):
    def test_initialization(self):

        dummy_config = {
            "output1": {"a": 10},
            "output2": {"b": 20},
            "config1": {"c": 30},
            "config2": {"d": [40]},
        }
        preprocessor = DummyPreprocessor(dummy_config)

        self.assertIn("output1", preprocessor.creates())
        self.assertIn("output2", preprocessor.creates())
        self.assertIn("output3", preprocessor.creates())
        self.assertIn("output4", preprocessor.creates())
        self.assertIn("output5", preprocessor.creates())
        self.assertIn("output6", preprocessor.creates())
        self.assertEqual(len(preprocessor.creates()), 6)
        out1 = preprocessor.get(None, "output1")
        self.assertEqual(out1["output1"], 10)
        self.assertEqual(out1["output2"], 20)
        self.assertEqual(len(out1), 2)
        out2 = preprocessor.get(None, "output2")
        self.assertEqual(out2["output1"], 10)
        self.assertEqual(out2["output2"], 20)
        self.assertEqual(len(out2), 2)
        out12 = preprocessor.get(None, "output1", "output2")
        self.assertEqual(out12["output1"], 10)
        self.assertEqual(out12["output2"], 20)
        self.assertEqual(len(out12), 2)
        out3 = preprocessor.get(None, "output3")
        self.assertEqual(out3["output3"], 70)
        self.assertEqual(len(out3), 1)
        out4 = preprocessor.get(None, "output4")
        self.assertTrue(out4["output4"])
        self.assertEqual(len(out4), 1)
        out6 = preprocessor.get(12, "output6")
        self.assertEqual(out6["output6"], 12)
        self.assertEqual(len(out6), 1)

        with self.assertRaises(KeyError):
            preprocessor.get(None, "non_existing_output")

    def test_deep_copy_config(self):
        dummy_config = {
            "config1": {"c": 30},
            "config2": {"d": [40]},
        }
        preprocessor = DummyPreprocessor(dummy_config)
        dummy_config["config2"]["d"][0] = 50
        out3 = preprocessor.get(None, "output3")
        self.assertEqual(out3["output3"], 70)

    def test_missing_config(self):
        dummy_config = {
            "config1": {"c": 7},
        }
        preprocessor = DummyPreprocessor(dummy_config)
        out5 = preprocessor.get(None, "output5")
        self.assertEqual(out5["output5"], 7)
        with self.assertRaises(KeyError):
            preprocessor.get(None, "output1")

    def test_overwrite_config(self):
        dummy_config = {
            "simple_output": {"a": 5},
        }
        preprocessor = SimpleDummyPreprocessor(dummy_config, a=15)
        out = preprocessor.get(None, "simple_output")
        self.assertEqual(out["simple_output"], 15)
    
    def test_no_config(self):
        preprocessor = SimpleDummyPreprocessor()
        out = preprocessor.get(None, "simple_output")
        self.assertIsNone(out["simple_output"])

    def test_overwrite_target_config(self):
        dummy_config = {
            "simple_output": {"a": 5},
        }
        preprocessor = SimpleDummyPreprocessor(dummy_config, overwrite_target="simple_output", a=25)
        out = preprocessor.get(None, "simple_output")
        self.assertEqual(out["simple_output"], 25)
    
    def test_overwrite_one_of_multiple_configs(self):
        dummy_config = {
            "output1": {"a": 10},
            "output2": {"b": 20},
            "config1": {"c": 30},
            "config2": {"d": [40]},
        }
        preprocessor = DummyPreprocessor(dummy_config, overwrite_target="config2", d=[100])
        out3 = preprocessor.get(None, "output3")
        self.assertEqual(out3["output3"], 130)
    
    def test_raises_on_overwrite_without_target_with_multiple_configs(self):
        dummy_config = {
            "output1": {"a": 10},
            "output2": {"b": 20},
            "config1": {"c": 30},
            "config2": {"d": [40]},
        }
        with self.assertRaises(ValueError):
            preprocessor = DummyPreprocessor(dummy_config, d=[100])

    def test_copy(self):
        dummy_config = {
            "simple_output": {"a": 5},
        }
        preprocessor1 = SimpleDummyPreprocessor(dummy_config)
        preprocessor2 = preprocessor1.copy(a=50)
        out1 = preprocessor1.get(None, "simple_output")
        out2 = preprocessor2.get(None, "simple_output")
        self.assertEqual(out1["simple_output"], 5)
        self.assertEqual(out2["simple_output"], 50)

    def test_from_function(self):
        def my_preprocessor(session, config):
            return session * config.get("multiplier", 1)

        preprocessor = OIDIQPreprocessor.from_function(
            my_preprocessor,
            creates="my_output",
            multiplier=3)

        out = preprocessor.get(10, "my_output")
        self.assertEqual(out["my_output"], 30)

    def test_from_function_no_config(self):
        def my_preprocessor(session):
            return session + 5

        preprocessor = OIDIQPreprocessor.from_function(
            my_preprocessor,
            creates="my_output_no_config")

        out = preprocessor.get(10, "my_output_no_config")
        self.assertEqual(out["my_output_no_config"], 15)
    
    def test_from_function_two_results(self):
        def my_preprocessor(session, config1, config2):
            return session + config1["a"], session + config2["b"]

        preprocessor = OIDIQPreprocessor.from_function(
            my_preprocessor,
            creates=("r1", "r2"))
        preprocessor.update_config("r1", a=5)
        preprocessor.update_config("r2", b=10)

        r1 = preprocessor.get(10, "r1")
        r2 = preprocessor.get(10, "r2")
        self.assertEqual(r1["r1"], 15)
        self.assertEqual(r2["r2"], 20)

    def test_rename_in_constructor(self):
        dummy_config = {
            "simple_output": {"a": 5},
        }
        preprocessor = SimpleDummyPreprocessor(dummy_config, overwrite_target="simple_output", name="renamed_output")
        out = preprocessor.get( None, "renamed_output")
        self.assertEqual(out["renamed_output"], 5)
    
    def test_rename_via_update_config(self):
        dummy_config = {
            "simple_output": {"a": 5},
        }
        preprocessor = SimpleDummyPreprocessor(dummy_config)
        out = preprocessor.get( None, "simple_output")
        self.assertEqual(out["simple_output"], 5)
        preprocessor.update_config(name="renamed_output")
        out = preprocessor.get( None, "renamed_output")
        self.assertEqual(out["renamed_output"], 5)
    
    def test_rename_mutiple_configs(self):
        dummy_config = {
            "output1": {"a": 10},
            "output2": {"b": 20},
        }
        preprocessor = DummyPreprocessor(dummy_config)
        out1 = preprocessor.get(None, "output1")
        self.assertEqual(out1["output1"], 10)
        out2 = preprocessor.get(None, "output2")
        self.assertEqual(out2["output2"], 20)
        preprocessor.update_config("output1", name="renamed_output1")
        preprocessor.update_config("output2", name="renamed_output2")
        out1 = preprocessor.get(None, "renamed_output1")
        self.assertEqual(out1["renamed_output1"], 10)
        out2 = preprocessor.get(None, "renamed_output2")
        self.assertEqual(out2["renamed_output2"], 20)

    def test_update_all_configs(self):
        dummy_config = {
            "out1": {"x": 5},
            "out2": {"x": 8},
        }
        preprocessor = MultiConfigPreprocessor(dummy_config)
        out1 = preprocessor.get(None, "out1")
        out2 = preprocessor.get(None, "out2")
        self.assertEqual(out1["out1"], 5)
        self.assertEqual(out2["out2"], 8 * 2)
        preprocessor.update_config("*", x=10)
        out1 = preprocessor.get(None, "out1")
        out2 = preprocessor.get(None, "out2")
        self.assertEqual(out1["out1"], 10)
        self.assertEqual(out2["out2"], 10 * 2)

    def test_update_all_config_constructor(self):
        dummy_config = {
            "out1": {"x": 5},
            "out2": {"x": 8},
        }
        preprocessor = MultiConfigPreprocessor(dummy_config, overwrite_target="*",
                                               x=12)
        out1 = preprocessor.get(None, "out1")
        out2 = preprocessor.get(None, "out2")
        self.assertEqual(out1["out1"], 12)
        self.assertEqual(out2["out2"], 12 * 2)


        


class DummyMetricCreator(OIDIQMetricCreator):
    @creates("metric1", "metric2")
    def create_metrics(self, session, config1, config2):
        metric_a = config1.create_quality_metric(raw_value=session, score=80)
        metric_b = config2.create_quality_metric(raw_value=config2["raw_score"], score=90)
        return metric_a, metric_b

class SimpleDummyMetricCreator(OIDIQMetricCreator):
    @creates("simple_metric")
    def simple_metric(self, session, config):
        metric = config.create_quality_metric(raw_value=config["x"], score=75)
        return metric

class TestOIDIQMetricCreator(unittest.TestCase):
    def test_initialization(self):
        dummy_config = {
            "metric1": {"name": "Metric A", "description": "First metric"},
            "metric2": {"name": "Metric B", "description": "Second metric", "raw_score": 0.75},
        }

        metric_creator = DummyMetricCreator(dummy_config)
        self.assertIn("Metric A", metric_creator.creates())
        self.assertIn("Metric B", metric_creator.creates())
        self.assertEqual(len(metric_creator.creates()), 2)
        result = metric_creator.get(0.85, "Metric A", "Metric B")
        self.assertIsInstance(result["Metric A"], QualityMetric)
        self.assertIsInstance(result["Metric B"], QualityMetric)
        self.assertEqual(result["Metric A"].name, "Metric A")
        self.assertEqual(result["Metric A"].raw_value, 0.85)
        self.assertEqual(result["Metric A"].score, 80)
        self.assertEqual(result["Metric A"].description, "First metric")
        self.assertEqual(result["Metric B"].name, "Metric B")
        self.assertEqual(result["Metric B"].raw_value, 0.75)
        self.assertEqual(result["Metric B"].score, 90)
        self.assertEqual(result["Metric B"].description, "Second metric")
    
    def test_from_function(self):
        def my_metric_creator(session, config):
            metric = config.create_quality_metric(raw_value=session * 2, score=75)
            return metric

        metric_creator = OIDIQMetricCreator.from_function(
            my_metric_creator,
            creates="My Metric",
            description="A test metric"
        )

        result = metric_creator.get(0.5, "My Metric")
        self.assertIsInstance(result["My Metric"], QualityMetric)
        self.assertEqual(result["My Metric"].name, "My Metric")
        self.assertEqual(result["My Metric"].raw_value, 1.0)
        self.assertEqual(result["My Metric"].score, 75)
        self.assertEqual(result["My Metric"].description, "A test metric")

    def test_from_function_no_description(self):
        def my_metric_creator(session, config):
            metric = config.create_quality_metric(raw_value=session * 2, score=75)
            return metric

        metric_creator = OIDIQMetricCreator.from_function(
            my_metric_creator,
            creates="My Metric"
        )
        result = metric_creator.get(0.5, "My Metric")
        self.assertIsInstance(result["My Metric"], QualityMetric)
        self.assertEqual(result["My Metric"].name, "My Metric")
        self.assertEqual(result["My Metric"].raw_value, 1.0)
        self.assertEqual(result["My Metric"].score, 75)
        self.assertEqual(result["My Metric"].description, "")

    def test_from_function_multiple_crates(self):
        def my_metric_creator(session, config1, config2):
            metric_a = config1.create_quality_metric(raw_value=session, score=80)
            metric_b = config2.create_quality_metric(raw_value=session + 0.1, score=85)
            return metric_a, metric_b

        metric_creator = OIDIQMetricCreator.from_function(
            my_metric_creator,
            creates=("Metric A", "Metric B"),
            description=("Description A", "Description B")

        )

        result = metric_creator.get(0.5, "Metric A", "Metric B")
        self.assertIsInstance(result["Metric A"], QualityMetric)
        self.assertIsInstance(result["Metric B"], QualityMetric)
        self.assertEqual(result["Metric A"].raw_value, 0.5)
        self.assertEqual(result["Metric A"].score, 80)
        self.assertEqual(result["Metric B"].raw_value, 0.6)
        self.assertEqual(result["Metric B"].score, 85)
        self.assertEqual(result["Metric A"].description, "Description A")
        self.assertEqual(result["Metric B"].description, "Description B")

    def test_raises_on_missing_description(self):
        def my_metric_creator(session, config1, config2):
            metric_a = config1.create_quality_metric(raw_value=session, score=80)
            metric_b = config2.create_quality_metric(raw_value=session + 0.1, score=85)
            return metric_a, metric_b

        with self.assertRaises(ValueError):
            metric_creator = OIDIQMetricCreator.from_function(
                my_metric_creator,
                creates=("Metric A", "Metric B"),
                description="Only one description"
            )

    def test_update_config(self):
        dummy_config = {
            "simple_metric": {"name": "Metric A", "x": 0.75},
        }

        metric_creator = SimpleDummyMetricCreator(dummy_config)
        result1 = metric_creator.get(0.5, "Metric A")
        self.assertEqual(result1["Metric A"].raw_value, 0.75)

        metric_creator.update_config(x=0.9)
        result2 = metric_creator.get(0.5, "Metric A")
        self.assertEqual(result2["Metric A"].raw_value, 0.9)

    def test_update_name(self):
        dummy_config = {
            "simple_metric": {"name": "Metric A", "x": 0.75},
        }

        metric_creator = SimpleDummyMetricCreator(dummy_config)
        result1 = metric_creator.get(0.5, "Metric A")
        self.assertEqual(result1["Metric A"].name, "Metric A")

        metric_creator.update_config(name="Updated Metric A")
        result2 = metric_creator.get(0.5, "Updated Metric A")
        self.assertEqual(result2["Updated Metric A"].name, "Updated Metric A")

    def test_add_name_postfix(self):
        dummy_config = {
            "metric1": {"name": "Metric A", "description": "First metric"},
            "metric2": {"name": "Metric B", "description": "Second metric", "raw_score": 0.75},
        }

        metric_creator = DummyMetricCreator(dummy_config)
        result1 = metric_creator.get(0.85, "Metric A", "Metric B")
        self.assertIn("Metric A", result1)
        self.assertIn("Metric B", result1)
        metric_creator.add_name_postfix("_v2")
        result2 = metric_creator.get(0.85, "Metric A_v2", "Metric B_v2")
        self.assertIn("Metric A_v2", result2)
        self.assertIn("Metric B_v2", result2)
        with self.assertRaises(KeyError):
            metric_creator.get(0.85, "Metric A")
        with self.assertRaises(KeyError):
            metric_creator.get(0.85, "Metric B")

    def test_add_name_postfix2(self):
        dummy_config = {
            "output1": {"a": 10},
            "output2": {"b": 20},
        }
        preprocessor = DummyPreprocessor(dummy_config)
        preprocessor.add_name_postfix("_new")
        out1 = preprocessor.get(None, "output1_new")
        self.assertEqual(out1["output1_new"], 10)
        out2 = preprocessor.get(None, "output2_new")
        self.assertEqual(out2["output2_new"], 20)

        out2 = preprocessor.get(13, "output6_new")
        self.assertEqual(out2["output6_new"], 13)
            
