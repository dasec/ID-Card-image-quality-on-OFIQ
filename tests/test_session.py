from typing import Tuple
import unittest

import numpy as np
from oidiq.utils import OIDIQConfig, QualityMetricConfig, QualityMetric, deep_copy_dict, creates, config
from oidiq.session import OIDIQPreprocessor, OIDIQSession, OIDIQMetricCreator, PreProcessors, OIDIQBatchSession


class TestOIDIQSession(unittest.TestCase):
    def test_session_initialization(self):
        session = OIDIQSession(np.array([[1, 2, 3], [4, 5, 6]]), [], [])
        self.assertTrue(np.array_equal(session.get_raw_image(), np.array([[1, 2, 3], [4, 5, 6]])))

    def test_load_from_file(self):
        session = OIDIQSession("tests/data/red.png", [], [])
        img = session.get_raw_image()
        self.assertEqual(img.shape, (3, 3, 3))
        self.assertEqual(np.sum(img[:, :, 0]), 255 * 3 * 3)
        self.assertEqual(np.sum(img[:, :, 1]), 0)
        self.assertEqual(np.sum(img[:, :, 2]), 0)

    def test_rgba_file_loading(self):
        session = OIDIQSession("tests/data/test_id.png", [], [])
        img = session.get_raw_image()
        self.assertEqual(img.shape[2], 3)

    def test_preprocess_getter(self):
        preprocessor = OIDIQPreprocessor.from_function(
            lambda session: tuple(range(7)),
            creates=(
                PreProcessors.CROPPED_IMAGE,
                PreProcessors.ID_CARD_CORNERS,
                PreProcessors.NORMALIZED_IMAGE,
                PreProcessors.NORMALIZED_FACE_MASK,
                PreProcessors.NORMALIZED_FACE_BACKGROUND_MASK,
                PreProcessors.NORMALIZED_FOREGROUND_MASK,
                PreProcessors.NORMALIZED_LUMINANCE,
            ),
        )
        hist_preprocessor = OIDIQPreprocessor.from_function(
            lambda session: 7,
            creates=PreProcessors.NORMALIZED_LUMINANCE_HISTOGRAM,
        )
        session = OIDIQSession(np.array([[0]]), [preprocessor, hist_preprocessor], [])

        self.assertEqual(session.get_cropped_image(), 0)
        self.assertEqual(session.get_id_card_corners(), 1)
        self.assertEqual(session.get_normalized_image(), 2)
        self.assertEqual(session.get_normalized_face_mask(), 3)
        self.assertEqual(session.get_normalized_face_background_mask(), 4)
        self.assertEqual(session.get_normalized_foreground_mask(), 5)
        self.assertEqual(session.get_normalized_luminance(), 6)
        self.assertEqual(session.get_normalized_luminance_histogram(), 7)

        with self.assertRaises(KeyError):
            session.get_preprocessed_image("non_existing_output")

    def test_result_hashing(self):
        def preprocess(session, config1, config2):
            if config1["counter"] > 0:
                raise ValueError("Should not be called again")
            config1["counter"] += 1
            return config1["counter"], 12

        preprocessor = OIDIQPreprocessor.from_function(
            preprocess,
            creates=(PreProcessors.CROPPED_IMAGE, PreProcessors.NORMALIZED_IMAGE),
            overwrite_target=PreProcessors.CROPPED_IMAGE,
            counter=0,
        )
        # session = OIDIQSession(np.array([[0]]), [preprocessor], [])
        session = OIDIQSession(np.array([[0]]), [preprocessor], [])
        self.assertEqual(session.get_cropped_image(), 1)
        self.assertEqual(session.get_cropped_image(), 1)
        self.assertEqual(session.get_normalized_image(), 12)

    def test_metric_creator(self):
        def create_metric(session, config):
            return config.create_quality_metric(raw_value=0.5, score=80)

        metric_creator = OIDIQMetricCreator.from_function(
            create_metric,
            creates="test_metric",
            name="test_metric",
            description="A test quality metric",
        )
        session = OIDIQSession(np.array([[0]]), [], [metric_creator])
        metric = session.get_score("test_metric")
        self.assertIsInstance(metric, QualityMetric)
        self.assertEqual(metric.name, "test_metric")
        self.assertEqual(metric.raw_value, 0.5)
        self.assertEqual(metric.score, 80)
        self.assertEqual(metric.description, "A test quality metric")

    def test_get_multiple_scores(self):
        def create_metric_a(session, config):
            return config.create_quality_metric(raw_value=0.3, score=70)

        def create_metric_b(session, config):
            return config.create_quality_metric(raw_value=0.8, score=90)

        metric_creator_a = OIDIQMetricCreator.from_function(
            create_metric_a,
            creates="metric_a",
            name="metric_a",
        )
        metric_creator_b = OIDIQMetricCreator.from_function(
            create_metric_b,
            creates="metric_b",
            name="metric_b",
        )
        session = OIDIQSession(np.array([[0]]), [], [metric_creator_a, metric_creator_b])
        metrics = session.get_scores("metric_a", "metric_b")
        self.assertEqual(len(metrics), 2)
        self.assertEqual(metrics[0].raw_value, 0.3)
        self.assertEqual(metrics[0].score, 70)
        self.assertEqual(metrics[1].raw_value, 0.8)
        self.assertEqual(metrics[1].score, 90)

    def test_get_scores_non_existing(self):
        def create_metric(session, config):
            return config.create_quality_metric(raw_value=0.5, score=80)

        metric_creator = OIDIQMetricCreator.from_function(
            create_metric,
            creates="existing_metric",
            name="existing_metric",
        )
        session = OIDIQSession(np.array([[0]]), [], [metric_creator])

        with self.assertRaises(KeyError):
            session.get_scores("non_existing_metric")

    def test_result_saving(self):
        call_count = {"count": 0}

        def create_metric(session, config):
            call_count["count"] += 1
            return config.create_quality_metric(raw_value=0.6, score=85)

        metric_creator = OIDIQMetricCreator.from_function(
            create_metric,
            creates="saved_metric",
            name="saved_metric",
        )
        session = OIDIQSession(np.array([[0]]), [], [metric_creator])
        metric1 = session.get_score("saved_metric")
        metric2 = session.get_score("saved_metric")
        self.assertEqual(call_count["count"], 1)
        self.assertIs(metric1, metric2)

    def test_get_all_scores(self):
        call_count = {"count": 0}

        def create_metrics(session, config1, config2):
            call_count["count"] += 1
            return (
                config1.create_quality_metric(raw_value=0.4, score=75),
                config2.create_quality_metric(raw_value=0.9, score=95),
            )

        metric_creator = OIDIQMetricCreator.from_function(
            create_metrics,
            creates=("metric1", "metric2"),
        )
        session = OIDIQSession(np.array([[0]]), [], [metric_creator])
        all_scores = session.get_all_scores()
        self.assertIn("metric1", all_scores)
        self.assertIn("metric2", all_scores)
        self.assertEqual(len(all_scores), 2)
        self.assertEqual(all_scores["metric1"].raw_value, 0.4)
        self.assertEqual(all_scores["metric1"].score, 75)
        self.assertEqual(all_scores["metric2"].raw_value, 0.9)
        self.assertEqual(all_scores["metric2"].score, 95)
        self.assertEqual(call_count["count"], 1)
        session.get_score("metric1")
        session.get_score("metric2")
        self.assertEqual(call_count["count"], 1)
        session.get_all_scores()
        self.assertEqual(call_count["count"], 1)

    def test_get_all_scores_complex(self):
        call_count = {"count_f1": 0, "count_f2": 0}

        class MetricCreator(OIDIQMetricCreator):
            @creates("metric1", "metric2")
            @config()
            def create_metric1(self, session: OIDIQSession) -> Tuple[QualityMetric, QualityMetric]:
                call_count["count_f1"] += 1
                return QualityMetric(name="metric1", raw_value=0.2, score=65), QualityMetric(
                    name="metric2", raw_value=0.4, score=75
                )

            @creates("metric3", "metric4")
            @config()
            def create_metric2(self, session: OIDIQSession) -> Tuple[QualityMetric, QualityMetric]:
                call_count["count_f2"] += 1
                return QualityMetric(name="metric3", raw_value=0.7, score=88), QualityMetric(
                    name="metric4", raw_value=0.5, score=80
                )

        session = OIDIQSession(np.array([[0]]), [], [MetricCreator()])
        all_scores = session.get_all_scores()
        self.assertIn("metric1", all_scores)
        self.assertIn("metric2", all_scores)
        self.assertIn("metric3", all_scores)
        self.assertIn("metric4", all_scores)
        self.assertEqual(len(all_scores), 4)
        self.assertEqual(all_scores["metric1"].raw_value, 0.2)
        self.assertEqual(all_scores["metric1"].score, 65)
        self.assertEqual(all_scores["metric2"].raw_value, 0.4)
        self.assertEqual(all_scores["metric2"].score, 75)
        self.assertEqual(all_scores["metric3"].raw_value, 0.7)
        self.assertEqual(all_scores["metric3"].score, 88)
        self.assertEqual(all_scores["metric4"].raw_value, 0.5)
        self.assertEqual(all_scores["metric4"].score, 80)
        self.assertEqual(call_count["count_f1"], 1)
        self.assertEqual(call_count["count_f2"], 1)

    def test_correct_naming(self):
        metric = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.55, score=82),
            creates="metric_name",
            name="Display Metric Name",
        )
        session = OIDIQSession(np.array([[0]]), [], [metric])
        result = session.get_score("Display Metric Name")

    def test_dependency_tree_empty(self):
        session = OIDIQSession(np.array([[0]]), [], [])
        tree = session._build_dependency_tree()
        self.assertIsInstance(tree, dict)
        self.assertEqual(tree, {"raw_image": {}})
        graph = session.dependency_graph(tree)
        self.assertIsInstance(graph, dict)
        self.assertEqual(graph, {"raw_image": (set(), set())})

    def test_dependency_tree_linear(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("a") * 2,
            creates="b",
        )
        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "a": {
                    "raw_image": {},
                }
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, {"b"}),
            "b": ({"a"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_branching(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() * 2,
            creates="b",
        )
        preprocessor_c = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("a") + session.get_preprocessed_image("b"),
            creates="c",
        )
        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b, preprocessor_c], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "raw_image": {},
            },
            "c": {
                "a": {
                    "raw_image": {},
                },
                "b": {
                    "raw_image": {},
                },
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a", "b"}),
            "a": ({"raw_image"}, {"c"}),
            "b": ({"raw_image"}, {"c"}),
            "c": ({"a", "b"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_with_metrics(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        metric_creator = OIDIQMetricCreator.from_function(
            lambda session, conf: conf.create_quality_metric(
                raw_value=session.get_preprocessed_image("a").sum(), score=90
            ),
            creates="metric",
            name="Metric",
        )
        metric_of_metric_creator = OIDIQMetricCreator.from_function(
            lambda session, conf: conf.create_quality_metric(
                raw_value=session.get_score("Metric").raw_value * 2, score=95
            ),
            creates="metric_of_metric",
        )
        session = OIDIQSession(np.array([[0]]), [preprocessor_a], [metric_creator, metric_of_metric_creator])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "score__Metric": {
                "a": {
                    "raw_image": {},
                }
            },
            "score__metric_of_metric": {
                "score__Metric": {
                    "a": {
                        "raw_image": {},
                    }
                }
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, {"score__Metric"}),
            "score__Metric": ({"a"}, {"score__metric_of_metric"}),
            "score__metric_of_metric": ({"score__Metric"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_multiple_roots(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: 17,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + session.get_preprocessed_image("a"),
            creates="b",
        )
        metric_c = OIDIQMetricCreator.from_function(
            lambda session, conf: conf.create_quality_metric(raw_value=0.5, score=90),
            creates="c",
            name="Metric C",
        )
        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b], [metric_c])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {},
            "b": {
                "raw_image": {},
                "a": {},
            },
            "score__Metric C": {},
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"b"}),
            "a": (set(), {"b"}),
            "b": ({"raw_image", "a"}, set()),
            "score__Metric C": (set(), set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_multiple_requests(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: 17,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image()
            + session.get_preprocessed_image("a")
            + session.get_raw_image()
            + session.get_preprocessed_image("a"),
            creates="b",
        )

        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {},
            "b": {
                "raw_image": {},
                "a": {},
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"b"}),
            "a": (set(), {"b"}),
            "b": ({"raw_image", "a"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_skip_layers(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("a") + session.get_raw_image(),
            creates="b",
        )
        preprocessor_c = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("b"),
            creates="c",
        )
        preprocessor_d = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("b"),
            creates="d",
        )
        preprocessor_e = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("c") + session.get_preprocessed_image("d"),
            creates="e",
        )
        preprocessor_f = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("e"),
            creates="f",
        )
        preprocessor_g = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("f") + session.get_preprocessed_image("e"),
            creates="g",
        )

        session = OIDIQSession(
            np.array([[0]]),
            [
                preprocessor_a,
                preprocessor_b,
                preprocessor_c,
                preprocessor_d,
                preprocessor_e,
                preprocessor_f,
                preprocessor_g,
            ],
            [],
        )
        tree = session._build_dependency_tree()
        raw_image_node = {"raw_image": {}}
        a_node = {"a": raw_image_node}
        b_node = {"b": {**raw_image_node, **a_node}}
        c_node = {"c": b_node}
        d_node = {"d": b_node}
        e_node = {"e": {**c_node, **d_node}}
        f_node = {"f": e_node}
        g_node = {"g": {**f_node, **e_node}}
        expected_tree = {
            **raw_image_node,
            **a_node,
            **b_node,
            **c_node,
            **d_node,
            **e_node,
            **f_node,
            **g_node,
        }
        expected_graph = {
            "raw_image": (set(), {"a", "b"}),
            "a": ({"raw_image"}, {"b"}),
            "b": ({"raw_image", "a"}, {"c", "d"}),
            "c": ({"b"}, {"e"}),
            "d": ({"b"}, {"e"}),
            "e": ({"c", "d"}, {"f", "g"}),
            "f": ({"e"}, {"g"}),
            "g": ({"e", "f"}, set()),
        }
        reqs_raw_image = session._list_requirements("raw_image")
        self.assertEqual(reqs_raw_image, [])
        reqs_a = session._list_requirements("a")
        self.assertEqual(reqs_a, ["raw_image"])
        reqs_b = session._list_requirements("b")
        self.assertCountEqual(reqs_b, ["a", "raw_image", "raw_image"])
        reqs_c = session._list_requirements("c")
        self.assertCountEqual(reqs_c, ["b", "a", "raw_image", "raw_image"])
        reqs_d = session._list_requirements("d")
        self.assertCountEqual(reqs_d, ["b", "a", "raw_image", "raw_image"])
        reqs_e = session._list_requirements("e")
        self.assertCountEqual(
            reqs_e, ["c", "b", "a", "raw_image", "raw_image", "d", "b", "a", "raw_image", "raw_image"]
        )
        reqs_f = session._list_requirements("f")
        self.assertCountEqual(
            reqs_f, ["e", "c", "b", "a", "raw_image", "raw_image", "d", "b", "a", "raw_image", "raw_image"]
        )
        reqs_g = session._list_requirements("g")
        self.assertCountEqual(
            reqs_g,
            [
                "f",
                "e",
                "c",
                "b",
                "a",
                "raw_image",
                "raw_image",
                "d",
                "b",
                "a",
                "raw_image",
                "raw_image",
                "e",
                "c",
                "b",
                "a",
                "raw_image",
                "raw_image",
                "d",
                "b",
                "a",
                "raw_image",
                "raw_image",
            ],
        )

        graph = session.dependency_graph(tree)
        self.assertEqual(graph, expected_graph)

        self.assertEqual(tree, expected_tree)

    def test_dependency_tree_skip_layers_simpler(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image(),
            creates="b",
        )
        preprocessor_c = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("a") + session.get_preprocessed_image("b"),
            creates="c",
        )
        preprocessor_d = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("b") + session.get_preprocessed_image("c"),
            creates="d",
        )

        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b, preprocessor_c, preprocessor_d], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "raw_image": {},
            },
            "c": {
                "a": {
                    "raw_image": {},
                },
                "b": {
                    "raw_image": {},
                },
            },
            "d": {
                "b": {
                    "raw_image": {},
                },
                "c": {
                    "a": {
                        "raw_image": {},
                    },
                    "b": {
                        "raw_image": {},
                    },
                },
            },
        }
        reqs_raw_image = session._list_requirements("raw_image")
        self.assertEqual(reqs_raw_image, [])
        reqs_a = session._list_requirements("a")
        self.assertEqual(reqs_a, ["raw_image"])
        reqs_b = session._list_requirements("b")
        self.assertEqual(reqs_b, ["raw_image"])
        reqs_c = session._list_requirements("c")
        self.assertCountEqual(reqs_c, ["a", "b", "raw_image", "raw_image"])
        reqs_d = session._list_requirements("d")
        self.assertCountEqual(reqs_d, ["a", "b", "b", "c", "raw_image", "raw_image", "raw_image"])

        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a", "b"}),
            "a": ({"raw_image"}, {"c"}),
            "b": ({"raw_image"}, {"c", "d"}),
            "c": ({"a", "b"}, {"d"}),
            "d": ({"b", "c"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_skip_layers_simpler2(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("a"),
            creates="b",
        )
        preprocessor_c = OIDIQPreprocessor.from_function(
            lambda session: session.get_preprocessed_image("b") + session.get_preprocessed_image("a"),
            creates="c",
        )

        session = OIDIQSession(np.array([[0]]), [preprocessor_a, preprocessor_b, preprocessor_c], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "a": {
                    "raw_image": {},
                }
            },
            "c": {
                "b": {
                    "a": {
                        "raw_image": {},
                    }
                },
                "a": {
                    "raw_image": {},
                },
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, {"b", "c"}),
            "b": ({"a"}, {"c"}),
            "c": ({"a", "b"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_one_class_two_preprocessors(self):
        class CreateAAndB(OIDIQPreprocessor):
            @creates("a")
            def create_a(self, session: OIDIQSession, config):
                raw = session.get_raw_image()
                return raw + 1
            
            @creates("b")
            def create_b(self, session: OIDIQSession, config):
                a = session.get_preprocessed_image("a")
                return a + 1
            
        
        session = OIDIQSession(np.array([[0]]), [CreateAAndB()], [])
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "a": {
                    "raw_image": {},
                }
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, {"b"}),
            "b": ({"a"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_batching_simple(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: [img + 1 for img in session.get_raw_image()],
            creates="a",
            batch_size=2
        )

        session = OIDIQBatchSession(
            np.array([[[[0]]], [[[10]]], [[[20]]]]),
            [preprocessor_a],
            [],
        )
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_batching_of_single_session(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: [img + 1 for img in session.get_raw_image()],
            creates="a",
            batch_size=2
        )

        session = OIDIQSession(
            np.array([[[[0]]], [[[10]]], [[[20]]]]),
            [preprocessor_a],
            [],
        )
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
        }
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, set()),
        }
        self.assertEqual(graph, expected_graph)

    def test_dependency_tree_batching(self):
        preprocessor_a = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="a",
        )
        preprocessor_b = OIDIQPreprocessor.from_function(
            lambda session: [a * 2 for a in session.get_preprocessed_image("a")],
            creates="b",
            batch_size=2
        )

        session = OIDIQBatchSession(
            np.array([[[[0]]], [[[10]]], [[[20]]]]),
            [preprocessor_a, preprocessor_b],
            [],
        )
        tree = session._build_dependency_tree()
        expected_tree = {
            "raw_image": {},
            "a": {
                "raw_image": {},
            },
            "b": {
                "a": {
                    "raw_image": {},
                }
            },
        }
        print(tree)
        self.assertEqual(tree, expected_tree)
        graph = session.dependency_graph(tree)
        expected_graph = {
            "raw_image": (set(), {"a"}),
            "a": ({"raw_image"}, {"b"}),
            "b": ({"a"}, set()),
        }
        self.assertEqual(graph, expected_graph)
    

    def test_batching(self):
        preprocessor = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="processed_image",
        )
        session = OIDIQBatchSession(
            np.array([[[[0]]], [[[10]]], [[[20]]]]),
            [preprocessor],
            [],
        )
        processed_images = session.get_preprocessed_image("processed_image")
        self.assertEqual(len(processed_images), 3)
        self.assertTrue(np.array_equal(processed_images[0], np.array([[[1]]])))
        self.assertTrue(np.array_equal(processed_images[1], np.array([[[11]]])))
        self.assertTrue(np.array_equal(processed_images[2], np.array([[[21]]])))

    def test_batching_preprocessor(self):
        shapes = []

        def preprocess(session):
            img = session.get_raw_image()
            shapes.append([i.shape for i in img])
            return [i+1 for i in img]

        preprocessor = OIDIQPreprocessor.from_function(
            preprocess,
            creates="processed_image",
            batch_size=2,
        )
        session = OIDIQBatchSession(
            np.array([[[[0]]], [[[10]]], [[[20]]]]),
            [preprocessor],
            [],
        )
        processed_images = session.get_preprocessed_image("processed_image")
        self.assertEqual(len(processed_images), 3)
        self.assertTrue(np.array_equal(processed_images[0], np.array([[[1]]])))
        self.assertTrue(np.array_equal(processed_images[1], np.array([[[11]]])))
        self.assertTrue(np.array_equal(processed_images[2], np.array([[[21]]])))

        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], [(1, 1, 1), (1, 1, 1)])
        self.assertEqual(shapes[1], [(1, 1, 1)])

    def test_batching_preprocessor_single_session(self):
        shapes = []

        def preprocess(session):
            img = session.get_raw_image()
            shapes.append([i.shape for i in img])
            return [i+1 for i in img]

        preprocessor = OIDIQPreprocessor.from_function(
            preprocess,
            creates="processed_image",
            batch_size=2,
        )
        session = OIDIQSession(
            np.array([[[0]]]),
            [preprocessor],
            [],
        )
        processed_image = session.get_preprocessed_image("processed_image")
        self.assertEqual(processed_image.shape, (1, 1, 1))
        self.assertTrue(np.array_equal(processed_image, np.array([[[1]]])))

        self.assertEqual(len(shapes), 1)
        self.assertEqual(shapes[0], [(1, 1, 1)])

    def test_batching_metric_creator(self):
        shapes = []

        def create_metric(session, config):
            img = session.get_raw_image()
            shapes.append([i.shape for i in img])
            return [config.create_quality_metric(raw_value=float(img[i].mean()), score=90) for i in range(len(img))]

        metric_creator = OIDIQMetricCreator.from_function(
            create_metric,
            creates="mean_intensity",
            name="Mean Intensity",
            batch_size=2,
        )
        session = OIDIQBatchSession(
            [np.array([[[0]]]), np.array([[[10]]]), np.array([[[20]]])],
            [],
            [metric_creator],
        )
        metric = session.get_score("Mean Intensity")
        self.assertEqual(len(metric), 3)
        self.assertEqual(metric[0].raw_value, 0.0)
        self.assertEqual(metric[1].raw_value, 10.0)
        self.assertEqual(metric[2].raw_value, 20.0)
        self.assertEqual(metric[0].score, 90)
        self.assertEqual(metric[1].score, 90)
        self.assertEqual(metric[2].score, 90)

        self.assertEqual(len(shapes), 2)
        self.assertEqual(shapes[0], [(1, 1, 1), (1, 1, 1)])
        self.assertEqual(shapes[1], [(1, 1, 1)])

    def test_batch_session_different_raw_img_shapes(self):
        session = OIDIQBatchSession(
            [np.array([[[0]]]), np.array([[[10, 10]]])],
            [],
            [],
        )

        raw_images = session.get_raw_image()
        self.assertEqual(len(raw_images), 2)
        self.assertTrue(np.array_equal(raw_images[0], np.array([[[0]]])))
        self.assertTrue(np.array_equal(raw_images[1], np.array([[[10, 10]]])))

    def test_register_preprocessor_after_initialization(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        preprocessor = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 5,
            creates="added_image",
        )
        session.register_preprocessor(preprocessor)

        added_image = session.get_preprocessed_image("added_image")
        self.assertTrue(np.array_equal(added_image, np.array([[5]])))

    def test_register_metric_creator_after_initialization(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        metric_creator = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=float(session.get_raw_image().mean()), score=100),
            creates="mean_intensity",
            name="Mean Intensity",
        )
        session.register_metric_creator(metric_creator)

        metric = session.get_score("Mean Intensity")
        self.assertEqual(metric.raw_value, 0.0)
        self.assertEqual(metric.score, 100)

    def test_register_preprocessor_overwrite(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        preprocessor1 = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 5,
            creates="added_image",
        )
        session.register_preprocessor(preprocessor1)

        preprocessor2 = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 10,
            creates="added_image",
            overwrite_target="added_image",
        )
        session.register_preprocessor(preprocessor2)

        added_image = session.get_preprocessed_image("added_image")
        self.assertTrue(np.array_equal(added_image, np.array([[10]])))

    def test_register_preprocessor_only_select_one(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        preprocessor1 = OIDIQPreprocessor.from_function(
            lambda session: (session.get_raw_image() + 5, session.get_raw_image() + 10),
            creates=("added_image_a", "added_image_b"),
        )
        preprocessor2 = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 20,
            creates="added_image_b",
            overwrite_target="added_image_b",
        )
        session.register_preprocessor(preprocessor2)
        session.register_preprocessor(preprocessor1, "added_image_a")
        added_image_a = session.get_preprocessed_image("added_image_a")
        added_image_b = session.get_preprocessed_image("added_image_b")
        self.assertTrue(np.array_equal(added_image_a, np.array([[5]])))
        self.assertTrue(np.array_equal(added_image_b, np.array([[20]])))

    def test_preprocessor_returns_unregistered_output(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        preprocessor = OIDIQPreprocessor.from_function(
            lambda session: (session.get_raw_image() + 5, session.get_raw_image() + 10),
            creates=("added_image_a", "added_image_b"),
        )
        session.register_preprocessor(preprocessor, "added_image_a")

        added_image_a = session.get_preprocessed_image("added_image_a")
        self.assertTrue(np.array_equal(added_image_a, np.array([[5]])))
        with self.assertRaises(KeyError):
            session.get_preprocessed_image("added_image_b")
    
    def test_get_all_scores_register_after(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        metric_creator = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.5, score=80),
            creates="test_metric",
            name="test_metric",
        )
        session.register_metric_creator(metric_creator)

        all_scores = session.get_all_scores()
        self.assertIn("test_metric", all_scores)
        self.assertEqual(len(all_scores), 1)
        self.assertEqual(all_scores["test_metric"].raw_value, 0.5)
        self.assertEqual(all_scores["test_metric"].score, 80)

    def test_get_all_scores_register_after_multiple(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        metric_creator_a = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.3, score=70),
            creates="metric_a",
            name="metric_a",
        )
        metric_creator_b = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.8, score=90),
            creates="metric_b",
            name="metric_b",
        )
        session.register_metric_creator(metric_creator_a)
        session.register_metric_creator(metric_creator_b)

        all_scores = session.get_all_scores()
        self.assertIn("metric_a", all_scores)
        self.assertIn("metric_b", all_scores)
        self.assertEqual(len(all_scores), 2)
        self.assertEqual(all_scores["metric_a"].raw_value, 0.3)
        self.assertEqual(all_scores["metric_a"].score, 70)
        self.assertEqual(all_scores["metric_b"].raw_value, 0.8)
        self.assertEqual(all_scores["metric_b"].score, 90)

    def test_get_all_scores_register_overwrite(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        metric_creator = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.5, score=80),
            creates="test_metric",
            name="test_metric",
        )
        session.register_metric_creator(metric_creator)

        metric_creator_overwrite = OIDIQMetricCreator.from_function(
            lambda session, config: config.create_quality_metric(raw_value=0.9, score=95),
            creates="test_metric",
            name="test_metric",
            overwrite_target="test_metric",
        )
        session.register_metric_creator(metric_creator_overwrite)

        all_scores = session.get_all_scores()
        self.assertIn("test_metric", all_scores)
        self.assertEqual(len(all_scores), 1)
        self.assertEqual(all_scores["test_metric"].raw_value, 0.9)
        self.assertEqual(all_scores["test_metric"].score, 95)

    def test_get_all_scores_returns_unregistered_output(self):
        session = OIDIQSession(np.array([[0]]), [], [])

        metric_creator = OIDIQMetricCreator.from_function(
            lambda session, config_a, config_b: (
                config_a.create_quality_metric(raw_value=0.5, score=80),
                config_b.create_quality_metric(raw_value=0.9, score=95),
            ),
            creates=("metric_a", "metric_b"),
        )
        session.register_metric_creator(metric_creator, "metric_a")

        all_scores = session.get_all_scores()
        self.assertIn("metric_a", all_scores)
        self.assertNotIn("metric_b", all_scores)
        self.assertEqual(len(all_scores), 1)
        self.assertEqual(all_scores["metric_a"].raw_value, 0.5)
        self.assertEqual(all_scores["metric_a"].score, 80)
    
    def test_batch_session_get_single_session(self):
        pre = OIDIQPreprocessor.from_function(
            lambda session: session.get_raw_image() + 1,
            creates="processed_image",
        )

        session = OIDIQBatchSession(
            np.array([[[[0]]], [[[10]]], [[[20]]], [[[30]]]]),
            [pre],
            [],
        )
        single_session = session[1]
        self.assertEqual(len(single_session.get_raw_image().shape), 3)
        processed_image = single_session.get_preprocessed_image("processed_image")
        self.assertEqual(processed_image.shape, (1, 1, 1))
        self.assertTrue(np.array_equal(processed_image, np.array([[[11]]])))

        batch_processes_image = session.get_preprocessed_image("processed_image")
        self.assertEqual(len(batch_processes_image), 4)
        self.assertTrue(np.array_equal(batch_processes_image[0], np.array([[[1]]])))
        self.assertTrue(np.array_equal(batch_processes_image[1], np.array([[[11]]])))
        self.assertTrue(np.array_equal(batch_processes_image[2], np.array([[[21]]])))
        self.assertTrue(np.array_equal(batch_processes_image[3], np.array([[[31]]])))


    
