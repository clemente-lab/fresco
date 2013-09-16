#!/usr/bin/env python
from __future__ import division

from unittest import TestCase, main
from fresco.parallel_processing import (multiprocess_functions,
                                        ProcessDefinition)

class ParallelProcessingTests(TestCase):
    """Tests for functions in the parallel_processing module."""

    def setUp(self):
        """Initialize data used in the tests."""
        def f(a, b):
            return a * b

        self.procs = [ProcessDefinition(f, (1, 2)),
                      ProcessDefinition(f, (3, 4))]
        self.results = []
        self.result_handler = self.results.append

    def test_multiprocess_functions(self):
        """Test running processes in parallel."""
        multiprocess_functions(self.procs, self.result_handler, 1)
        self.assertEqual(sorted(self.results), [2, 12])


class ProcessDefinitionTests(TestCase):
    """Tests for the ProcessDefinition class."""

    def setUp(self):
        """Initialize data used in the tests."""
        def function_no_args():
            return 42

        def function_pos_args(a, b):
            return a * b

        def function_kw_args(a=4, b=2, c=10):
            return a * b * c

        def function_mixed_args(a, b, c=10):
            return (a + b) * c

        self.pd_no_args = ProcessDefinition(function_no_args)
        self.pd_pos_args = ProcessDefinition(function_pos_args, (4, 2))
        self.pd_kw_args = ProcessDefinition(function_kw_args,
                                            keyword_arguments={'c': 3})
        self.pd_tag = ProcessDefinition(function_mixed_args,
                                        positional_arguments=(5, 6),
                                        keyword_arguments={'c': 7}, tag='foo')

    def test_process(self):
        """Test running a process defined in a ProcessDefinition instance."""
        obs = self.pd_no_args.process()
        self.assertEqual(obs, 42)

        obs = self.pd_pos_args.process()
        self.assertEqual(obs, 8)

        obs = self.pd_kw_args.process()
        self.assertEqual(obs, 24)

    def test_process_with_tag(self):
        """Test running a process with a tag."""
        obs = self.pd_tag.process()
        self.assertEqual(obs, ('foo', 77))


if __name__ == '__main__':
    main()
