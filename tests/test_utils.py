#!/usr/bin/env python
from __future__ import division

from unittest import TestCase, main
from sklearn.linear_model import LogisticRegression
from fresco.utils import (check_input_type, get_list_accuracy, InputTypeError,
                          parse_model_string, parse_object_string_sample, normalized_scores)
import numpy as np
    
class UtilsTests(TestCase):
    """Tests for functions in the utils module."""

    def setUp(self):
        """Initialize data used in the tests."""
        pass

    def test_parse_object_string_sample(self):
        """Test parsing sample name from object string."""
        obs = parse_object_string_sample('foo_bar123')
        self.assertEqual(obs, 'foo')

        with self.assertRaises(ValueError):
            _ = parse_object_string_sample('foo')

    def test_parse_model_string(self):
        """Test parsing a model name string."""
        obs = parse_model_string('lr')
        self.assertTrue(isinstance(obs, LogisticRegression))

        with self.assertRaises(InputTypeError):
            _ = parse_model_string('foo')

    def test_get_list_accuracy(self):
        """Test computing accuracy between two lists."""
        # All match.
        obs = get_list_accuracy([1, 2, 3], [1, 2, 3])
        self.assertEqual(obs, 1.0)

        # Some match.
        obs = get_list_accuracy([1, 2, 3], [1, 3, 3])
        self.assertEqual(obs, 2 / 3)

        # No matches.
        obs = get_list_accuracy([1, 2, 3], [3, 0, 1])
        self.assertEqual(obs, 0.0)

        # Single element lists.
        obs = get_list_accuracy([42], [42])
        self.assertEqual(obs, 1.0)

        # Empty lists.
        with self.assertRaises(ValueError):
            _ = get_list_accuracy([], [])

        with self.assertRaises(ValueError):
            _ = get_list_accuracy([1, 2, 3], [1])

    def test_check_input_type(self):
        """Test validating the type of an input."""
        # Single acceptable type.
        obs = check_input_type('foo', 42, int)
        self.assertEqual(obs, None)

        # Multiple acceptable types.
        obs = check_input_type('foo', 42, (str, int))
        self.assertEqual(obs, None)

        # Invalid type (single option).
        with self.assertRaises(InputTypeError):
            _ = check_input_type('foo', 42, float)

        # Invalid type (multiple options).
        with self.assertRaises(InputTypeError):
            _ = check_input_type('foo', 42, (str, float))
            
    def test_normalized_scores(self):
        """Test normalizing scores"""
        scores = np.array([0.5, None, 2, 2, -1, -0.5, None])
        exclude_list = np.array([2, 1])
        
        norm_scores = normalized_scores(scores, exclude_list)
        
        expected = np.array([0.3, 0.0, 0.0, 0.6, 0.0, 0.1, 0.0])
        
        self.assertEquals(len(norm_scores), len(expected))
        for i in range(len(norm_scores)):
            self.assertEqual(norm_scores[i], expected[i])

if __name__ == '__main__':
    main()
