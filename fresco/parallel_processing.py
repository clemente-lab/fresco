import multiprocessing

def multiprocess_functions(process_definitions, result_handler, n_procs):
    #special case for serial execution
    if n_procs == 0:
        for process_definition in process_definitions:
            result_handler(process_definition.process())
        return
    
    processes = []
    result_queue = multiprocessing.Queue()

    def function_wrapper(process_definition, result_queue):
        outcome = process_definition.process()
        result_queue.put_nowait(outcome)

    def wait_for_finished(n_survivers):
        while len(processes) > n_survivers:
            while not result_queue.empty():
                res = result_queue.get()
                result_handler(res)

            for pn in range(len(processes)):
                process = processes[pn]
                if not process.is_alive():
                    processes.pop(pn)
                break

    for process_definition in process_definitions:
        wait_for_finished(n_procs)

        process = multiprocessing.Process(target = function_wrapper, args = 
                                          (process_definition, result_queue) )
        processes.append(process)
        process.start()

    wait_for_finished(0)

class ProcessDefinition(object):
    def __init__(self, function, positional_arguments=None, keyword_arguments=None, tag=None):
        self.function = function
        if positional_arguments:
            self.positional_arguments = positional_arguments
        else:
            self.positional_arguments = ()
        if keyword_arguments:
            self.keyword_arguments = keyword_arguments
        else:
            self.keyword_arguments = {}
        self.tag = tag
        
    def process(self):
        result = self.function(*self.positional_arguments, **self.keyword_arguments)
        if self.tag is not None:
            return self.tag, result
        else:
            return result
