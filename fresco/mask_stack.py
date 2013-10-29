import numpy as np

class MaskStack:
    """
    An abstraction for a 'stack' of masks, where the aggregate mask is equivalent
    to applying each mask in turn.

    Masks are pushed on to the stack in the form of a list of indices to be excluded
    from the currently masked array.
    The aggregate mask is returned as a boolean array, where True values have not
    been excluded by any mask in the stack.
    
    IMPORTANT NOTE:
    This module uses index masks to denote which elements to EXCLUDE, while
    the semantics for NumPy ( arr[mask] ) denote which elements to INCLUDE.
    I've found the former model to be simpler to use in a stack.
    
    When using library functions to generate masks, like
    sklearn.cross_validation.KFolds, you can simply switch the test mask and
    the train mask when using this module vs. simple NumPy masks. For example:
    
    samples = np.array( [1, 2, 3, 4, 5] )
    test_mask = np.array( [1, 2] ) # indices to include while testing
    train_mask = np.array( [3, 4, 5] ) #indices to include while training
    
    test_samples_np = samples[test_mask]
    train_samples_np = samples[train_mask]
    
    mask_stack = MaskStack( samples.shape[0] )
    mask_stack.push_mask( train_mask ) #exclude the train samples
    test_samples_ms = samples[mask_stack.get_aggregate_mask()]
    mask_stack.pop()
    mask_stack.push_mask( test_mask ) #exclude the test samples
    train_samples_ms = samples[mask_stack.get_aggregate_mask()]
    mask_stack.pop_mask()
    
    # test_samples_np == test_samples_ms
    # train_samples_np == train_samples_ms
    """

    def __init__(self, length):
        """
        Initialize the MaskStack to produce a mask for an array of length 'length'
        """
        self.agg_mask = np.array([True] * length)
        self.reference_indices = np.array(range(length))
        self.exclusion_indices_stack = []

    def push_mask(self, relative_exclusion_indices):
        """
        Add an index mask to the stack and update the aggregate mask.
        
        Masks should be passed as a list of indices to be masked (set to False
        in aggregate mask), where each index indexes into the current length 
        of the array.
        For example, say that current aggregate mask has 10 True values, so
        a masked array would be of length 10. To remove the 3rd and 5th
        elements of the masked array, we would do:
        
        push_stack( array([2, 4]) ) 
        
        The aggregate mask after this call would have 8 True values, the
        3rd and 5th previously True values now set to False.
        """

        #find the currently unmasked indices
        current_indices = self.reference_indices[self.agg_mask]
        #index into the unmasked indices to find the absolute locations of the new masked values
        absolute_exclusion_indices = current_indices[relative_exclusion_indices]
        
        #add the absolute indices to the stack and update the aggregate mask
        self.exclusion_indices_stack.append(absolute_exclusion_indices)
        self.agg_mask[absolute_exclusion_indices] = False

    def pop_mask(self):
        """
        Remove the most recent mask from the stack and update the aggregate mask.
        """

        mask = self.exclusion_indices_stack.pop()
    
        self.agg_mask[mask] = True

    def get_aggregate_mask(self):
        """
        Return the aggregate boolean mask of the stack, in which the indices
        that have not been masked are True.
        """

        return self.agg_mask
