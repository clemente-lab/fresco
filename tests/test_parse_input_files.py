#!/usr/bin/env python
from __future__ import division

from unittest import TestCase, main
from fresco.parse_input_files import (FeatureMapFileFormatError,
                                      MappingFileFormatError, read_split_file,
                                      read_mapping_file)
from fresco.utils import InputTypeError

class ParseInputFilesTests(TestCase):
    """Tests for functions in the parse_input_files module."""

    def setUp(self):
        """Initialize data used in the tests."""
        self.split1_lines = split1_str.split('\n')
        self.split_dup_groups_lines = split_dup_groups_str.split('\n')
        self.split_dup_objs_lines = split_dup_objs_str.split('\n')

        self.map1_lines = map1_str.split('\n')
        self.map_empty_cells_lines = map_empty_cells_str.split('\n')
        self.map_dup_samples_lines = map_dup_samples_str.split('\n')

    def test_read_split_file(self):
        """Test parsing a feature-to-object map."""
        exp = ({'F1': ['A', 'B'], 'F2': ['C'], 'F3': ['D', 'E', 'F']},
               {'A': 'F1', 'C': 'F2', 'B': 'F1', 'E': 'F3', 'D': 'F3',
                'F': 'F3'})
        obs = read_split_file(self.split1_lines)
        self.assertEqual(obs, exp)

    def test_read_split_file_invalid_input(self):
        """Test parsing invalid input raises an error."""
        with self.assertRaises(FeatureMapFileFormatError):
            _ = read_split_file(self.split_dup_groups_lines)

        with self.assertRaises(FeatureMapFileFormatError):
            _ = read_split_file(self.split_dup_objs_lines)

    def test_read_mapping_file(self):
        """Test parsing a mapping file into a nested dictionary."""
        exp = {'A': {'Foo': 'f1', 'Bar': 'b1'},
               'B': {'Foo': 'f2', 'Bar': 'b2'}}
        obs = read_mapping_file(self.map1_lines)
        self.assertEqual(obs, exp)

    def test_read_mapping_file_invalid_input(self):
        """Test parsing invalid input raises an error."""
        with self.assertRaises(MappingFileFormatError):
            _ = read_mapping_file(self.map_empty_cells_lines)

        with self.assertRaises(MappingFileFormatError):
            _ = read_mapping_file(self.map_dup_samples_lines)


split1_str = """F1\tA\tB
F2\tC
F3\tD\tE\tF
"""

split_dup_groups_str = """F1\tA\tB
F2\tC
F1\tD
"""

split_dup_objs_str = """F1\tA\tB
F2\tC
F3\tD\tB
"""

map1_str = """#SampleID\tFoo\tBar
A\tf1\tb1
B\tf2\tb2
"""

map_empty_cells_str = """#SampleID\tFoo\tBar
A\tf1\tb1
B\tb2
"""

map_dup_samples_str = """#SampleID\tFoo\tBar
A\tf1\tb1
B\tf2\tb2
A\tf3\tb3
"""


if __name__ == '__main__':
    main()
