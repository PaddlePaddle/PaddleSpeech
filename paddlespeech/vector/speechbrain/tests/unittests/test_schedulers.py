def test_NewBobScheduler():

    from speechbrain.nnet.schedulers import NewBobScheduler

    scheduler = NewBobScheduler(initial_value=0.8)

    prev_lr, next_lr = scheduler(1.0)
    assert prev_lr == 0.8
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.4

    prev_lr, next_lr = scheduler(0.5)
    assert next_lr == 0.4

    scheduler = NewBobScheduler(initial_value=0.8, patient=3)
    prev_lr, next_lr = scheduler(1.0)
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    prev_lr, next_lr = scheduler(1.1)
    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.8

    prev_lr, next_lr = scheduler(1.1)
    assert next_lr == 0.4
    assert scheduler.current_patient == 3
