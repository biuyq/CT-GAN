import tflib as lib
import tflib.debug

import numpy as np
import tensorflow as tf

import itertools
import time
import collections
import os
import locale

locale.setlocale(locale.LC_ALL, '')

SAVE_FILE = "params_latest.ckpt"

def train_loop(
    session,
    inputs,
    cost,
    train_data,
    times,
    prints=[],
    test_data=None,
    callback=None,
    optimizer=tf.train.AdamOptimizer(),
    inject_total_iters=False,
    debug_mode=False,
    save_params=False,
    profile=False
    ):
    saver = tf.train.Saver()

    prints = [('cost', cost)] + prints

    grads_and_vars = optimizer.compute_gradients(
        cost,
        colocate_gradients_with_ops=True
    )

    print "Params:"
    total_param_count = 0
    for g, v in grads_and_vars:
        shape = v.get_shape()
        shape_str = ",".join([str(x) for x in v.get_shape()])

        param_count = 1
        for dim in shape:
            param_count *= int(dim)
        total_param_count += param_count

        if g == None:
            print "\t{} ({}) [no grad!]".format(v.name, shape_str)
        else:
            print "\t{} ({})".format(v.name, shape_str)
    print "Total param count: {}".format(
        locale.format("%d", total_param_count, grouping=True)
    )

    for i in xrange(len(grads_and_vars)):
        g, v = grads_and_vars[i]
        if g == None:
            grads_and_vars[i] = (tf.zeros_like(v), v)
        else:
            grads_and_vars[i] = (tf.clip_by_value(g, -1., 1.), v)            

    train_op = optimizer.apply_gradients(grads_and_vars)

    def train_fn(input_vals, i, profile=False):
        if i==10 and profile:
            run_metadata = tf.RunMetadata()
            result = session.run(
                [p[1] for p in prints] + [train_op],
                feed_dict={sym:real for sym, real in zip(inputs, input_vals)},
                options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE),
                run_metadata=run_metadata
            )[:-1]

            from tensorflow.python.client import timeline
            trace = timeline.Timeline(step_stats=run_metadata.step_stats)
            trace_file = open('timeline.ctf.json', 'w')
            trace_file.write(trace.generate_chrome_trace_format())
            print "Profiled!"
        else:
            return session.run(
                [p[1] for p in prints] + [train_op],
                feed_dict={sym:real for sym, real in zip(inputs, input_vals)}
            )[:-1]

    def eval_fn(input_vals):
        return session.run(
            [p[1] for p in prints],
            feed_dict={sym:real for sym, real in zip(inputs, input_vals)}
        )

    def print_stats_fn(input_vals):
        if debug_mode:
            lib.debug.print_all_stats(
                feed_dict={sym:real for sym, real in zip(inputs, input_vals)}
            )
    
    print "Initializing variables"
    session.run(tf.initialize_all_variables())

    total_iters = 0
    total_seconds = 0.
    last_print = 0
    last_test = 0
    last_gen = 0
    all_outputs = []
    all_stats = []
    run_times = []

    print "Training!"
    for epoch in itertools.count():
        generator = train_data()
        while True:
            try:
                input_vals = generator.next()
            except StopIteration:
                break

            if inject_total_iters:
                input_vals = [np.int32(total_iters)] + list(input_vals)

            print_stats_fn(input_vals)

            start_time = time.time()
            outputs = train_fn(input_vals, total_iters, profile)
            run_time = time.time() - start_time
            total_seconds += run_time
            total_iters += 1
            run_times.append(run_time)

            all_outputs.append(outputs)

            if (times['mode']=='iters' and total_iters-last_print == times['print_every']) or \
                (times['mode']=='seconds' and total_seconds-last_print >= times['print_every']):

                mean_outputs = np.array(all_outputs).mean(axis=0)

                test = (test_data is not None) and \
                    ( \
                        (times['mode']=='iters' and total_iters-last_test == times['test_every']) or \
                        (times['mode']=='seconds' and total_seconds-last_test >= times['test_every'])
                    )


                if test:

                    if inject_total_iters:
                        test_outputs = [
                            eval_fn([np.int32(total_iters)] + list(input_vals)) 
                            for input_vals in test_data()
                        ]
                    else:
                        test_outputs = [
                            eval_fn(input_vals) 
                            for input_vals in test_data()
                        ]

                    test_mean_outputs = np.array(test_outputs).mean(axis=0)

                stats = collections.OrderedDict()
                stats['epoch'] = epoch
                stats['iters'] = total_iters
                for i,p in enumerate(prints):
                    stats['train '+p[0]] = mean_outputs[i]
                if test:
                    for i,p in enumerate(prints):
                        stats['test '+p[0]] = test_mean_outputs[i]
                stats['secs'] = total_seconds
                stats['secs/iter'] = np.mean(run_times)

                print_str = ""
                for k,v in stats.items():
                    if isinstance(v, int):
                        print_str += "{}:{}\t".format(k,v)
                    else:
                        print_str += "{}:{:.4f}\t".format(k,v)
                print print_str[:-1] # omit the last \t

                all_stats.append(stats)

                all_outputs = []
                run_times = []
                last_print += times['print_every']

                if test:
                    last_test += times['test_every']

            if callback:
                if (times['mode']=='iters' and total_iters-last_gen==times['callback_every']) or \
                    (times['mode']=='seconds' and total_seconds-last_gen >= times['callback_every']):

                    tag = "iters{}_time{}".format(total_iters, total_seconds)
                    if callback is not None:
                        callback(tag)

                    if save_params:
                        saver.save(session, SAVE_FILE)
                        print "Saved params to {}".format(SAVE_FILE)

                    last_gen += times['callback_every']

            if (times['mode']=='iters' and total_iters == times['stop_after']) or \
                (times['mode']=='seconds' and total_seconds >= times['stop_after']):

                print "Done!"

                try: # This only matters on Ishaan's computer
                    import experiment_tools
                    experiment_tools.send_sms("done!")
                except ImportError:
                    pass

                return all_stats