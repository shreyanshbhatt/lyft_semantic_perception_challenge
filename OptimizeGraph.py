import tensorflow as tf
from tensorflow.python.framework import graph_util as tf_graph_util
import shutil

if __name__ == '__main__':
    tf.reset_default_graph()
    gd = tf.GraphDef()
    output_graph_file = "frozen_model/graph_optimized.pb"
    with tf.gfile.Open(output_graph_file, 'rb') as f:
        gd.ParseFromString(f.read())
    tf.import_graph_def(gd, name='')
    print("{} ops in the optimised graph".format(len(gd.node)))

    # save model in same format as usual
    #shutil.rmtree(a, ignore_errors=True)
    #if not os.path.exists(args.optimised_model_dir):
    #    os.makedirs(args.optimised_model_dir)

    print('saving optimised model as saved_model to {optimized_model}')
    #model = fcn8vgg16.FCN8_VGG16(define_graph=False)
    tf.reset_default_graph()
    tf.import_graph_def(gd, name='')
    with tf.Session() as sess:
        builder = tf.saved_model.builder.SavedModelBuilder('optimized_model')
        builder.add_meta_graph_and_variables(sess, ['FCNVGG'])
        builder.save()
    shutil.move('frozen_model/graph_optimized.pb', 'optimized_model')

