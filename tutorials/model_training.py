import trainer.lib.demo_data as demo_data

if __name__ == '__main__':
    data_path = r'C:\Users\rapha\Desktop\data'
    sd = demo_data.SourceData('D:\\')
    ds = demo_data.build_mnist(data_path, sd)

    class_name = 'digit'  # The structure that we now train for

    BATCH_SIZE = 32
    EPOCHS = 60

    # N = ds.get_subject_count(split='train')

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

    # visboard = ml.VisBoard(run_name=lib.create_identifier('test'))
    # seg_network = ml.seg_network.SegNetwork("SegNetwork", 3, 2, ds, batch_size=BATCH_SIZE, vis_board=visboard)
    # seg_network.print_summary()

    # test_loader = data.DataLoader(seg_network.get_torch_dataset(split='test', mode=ModelMode.Eval),
    #                               batch_size=BATCH_SIZE, shuffle=True)
    # machine_loader = data.DataLoader(seg_network.get_torch_dataset(split='machine', mode=ModelMode.Usage),
    #                                  batch_size=BATCH_SIZE,
    #                                  shuffle=True)

    # seg_network.train_supervised(
    #     'bone',
    #     train_split='train',
    #     max_epochs=50,
    #     load_latest_state=True)
    # save_predictions('./out2/', split='train')
    # save_predictions('./out2/', split='test')
    # save_predictions('./out2/', split='machine')
