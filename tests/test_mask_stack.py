#!/usr/bin/env python
from __future__ import division
from unittest import TestCase, main
from fresco.mask_stack import MaskStack
import numpy as np

class MaskStackTests(TestCase):
    """Tests for functions in the mask_stack module."""

    def setUp(self):
        """Initialize data used in the tests."""
        self.length = 100
        self.unmasked_arr = np.array(range(self.length))
        self.mask_stack = MaskStack(self.length)

    def test_get_aggregate_mask(self):
        """
        Test generating the aggregate mask.
        """
        
        def test_masked_arr(expected_arr):
            mask = self.mask_stack.get_aggregate_mask()
            self.assertEqual(len(mask), self.length)
            masked_arr = self.unmasked_arr[mask]
            #I don't seem to be able to check equality of two arrays
            self.assertEqual(list(masked_arr), list(expected_arr))
        
        #the mask stack shouldn't do anything before we push onto it
        test_masked_arr(self.unmasked_arr)

        #push a mask, make sure the result is what we expect
        first_mask = np.array(range(0, 100, 5))
        self.mask_stack.push_mask(first_mask)
        first_expected_arr = np.array(
            [self.unmasked_arr[i] for i in range(self.length) if not i in first_mask])
        test_masked_arr(first_expected_arr)

        #push another mask, check the result
        second_mask = np.array(range(0, len(first_expected_arr), 3))
        self.mask_stack.push_mask(second_mask)
        second_expected_arr = np.array(
            [first_expected_arr[i] for i in range(len(first_expected_arr)) if not i in second_mask])
        test_masked_arr(second_expected_arr)

        #pop off the second mask, should be the same as when we added the first
        self.mask_stack.pop_mask()
        test_masked_arr(first_expected_arr)
        
        #pop off the first mask, should be an empty stack again
        self.mask_stack.pop_mask()
        test_masked_arr(self.unmasked_arr)
