import numpy as np

class MaskStack:
    """
    An abstraction for a 'stack' of masks, where the aggregate mask is equivalent
    to applying each mask in turn.

    Masks are pushed on to the stack in the form of a list of indeces to be excluded
    from the currently masked array.
    The aggregate mask is returned as a boolean array, where True values have not
    been excluded by any mask in the stack.
    """

    def __init__(self, length):
        """
        Initialize the MaskStack to produce a mask for an array of length 'length'
        """
        self.agg_mask = np.array([True] * length)
        self.reference_indeces = np.array(range(length))
        self.exclusion_indeces_stack = []

    def push_mask(self, relative_exclusion_indeces):
        """
        Add an index mask to the stack and update the aggregate mask.
        
        Masks should be passed as a list of indeces to be masked (set to False
        in aggregate mask), where each index indexes into the current length 
        of the array.
        For example, say that current aggregate mask has 10 True values, so
        a masked array would be of length 10. To remove the 3rd and 5th
        elements of the masked array, we would do:
        
        push_stack( array([2, 4]) ) 
        
        The aggregate mask after this call would have 8 True values, the
        3rd and 5th previously True values now set to False.
        """

        #find the currently unmasked indeces
        current_indeces = self.reference_indeces[self.agg_mask]
        #index into the unmasked indeces to find the absolute locations of the new masked values
        absolute_exclusion_indeces = current_indeces[relative_exclusion_indeces]
        
        #add the absolute indeces to the stack and update the aggregate mask
        self.exclusion_indeces_stack.append(absolute_exclusion_indeces)
        self.agg_mask[absolute_exclusion_indeces] = False

    def pop_mask(self):
        """
        Remove the most recent mask from the stack and update the aggregate mask.
        """

        mask = self.exclusion_indeces_stack.pop()
    
        self.agg_mask[mask] = True

    def get_aggregate_mask(self):
        """
        Return the aggregate boolean mask of the stack, in which the indecies
        that have not been masked are True.
        """

        return self.agg_mask
