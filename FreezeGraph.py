import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util

if __name__ == '__main__':
    checkpoint = tf.train.get_checkpoint_state("./model10_dknow/")
    input_checkpoint = checkpoint.model_checkpoint_path
    print("freezing from {}".format(input_checkpoint))
    saver = tf.train.import_meta_graph(input_checkpoint + '.meta', clear_devices=True)
    graph = tf.get_default_graph()
    input_graph_def = graph.as_graph_def()
    print("{} ops in the input graph".format(len(input_graph_def.node)))

    output_node_names = "prediction_softmax"

    # freeze graph
    with tf.Session() as sess:
        saver.restore(sess, input_checkpoint)
        # use a built-in TF helper to export variables to constants
        output_graph_def = tf_graph_util.convert_variables_to_constants(
            sess,
            input_graph_def,
            output_node_names.split(",")
        )
    print("{} ops in the frozen graph".format(len(output_graph_def.node)))
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder('frozen_model_2')
        builder.add_meta_graph_and_variables(sess, ['FCNVGG'])
        builder.save()

    with tf.gfile.GFile('frozen_model_2/graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
'''
    if os.path.exists(args.frozen_model_dir):
        shutil.rmtree(args.frozen_model_dir)

    # save model in same format as usual
    print('saving frozen model as saved_model to {}'.format(args.frozen_model_dir))
    model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(output_graph_def, name='')
    with tf.Session() as sess:
        model.save_model(sess, args.frozen_model_dir)

    print('saving frozen model as graph.pb (for transforms) to {}'.format(args.frozen_model_dir))
    with tf.gfile.GFile(args.frozen_model_dir+'/graph.pb', "wb") as f:
        f.write(output_graph_def.SerializeToString())
'''

