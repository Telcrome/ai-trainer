import trainer.lib as lib

if __name__ == '__main__':
    tracker = lib.ProgressTracker(run_desc='')
    tracker.add_result(result_name='3ba', flag='success')
    tracker.add_result(result_name='3bac', flag='fail')
    tracker.add_result(result_name='3baawe', flag='fail')
    tracker.add_result(result_name='3baawe', flag='fail')

    print(tracker)
