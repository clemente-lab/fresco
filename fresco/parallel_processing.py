import multiprocessing

def multiprocess_functions(functions, result_handler, n_procs):
    processes = []
    result_queue = multiprocessing.Queue()

    def function_wrapper(function, function_args, result_queue):
        outcome = function(*function_args)
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
        

    for i in range(len(functions)):
        wait_for_finished(n_procs)

        process = multiprocessing.Process(target = function_wrapper, args = 
                                          (functions[i][0], functions[i][1], result_queue) )
        processes.append(process)
        process.start()

    wait_for_finished(0)
