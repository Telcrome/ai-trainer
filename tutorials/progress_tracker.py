import trainer.lib as lib

if __name__ == '__main__':
    # lib.reset_complete_database()
    sess = lib.Session()

    tracker = lib.Experiment.build_new('Experiment Demo')
    sess.add(tracker)
    # tracker = lib.Experiment(exp_name='')
    tracker.add_result(result_name='3ba', flag='success')
    tracker.add_result(result_name='3bac', flag='fail')
    tracker.add_result(result_name='3baawe', flag='fail')
    tracker.add_result(result_name='3baawe', flag='fail')

    # print(tracker)
    sess.commit()

    print(tracker.get_results('fail'))
    print(lib.Experiment.get_all_with_flag(sess, 'Experiment Demo', flag='fail'))
