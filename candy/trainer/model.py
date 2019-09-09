import shutil
import numpy as np
import tensorflow as tf

tf.logging.set_verbosity(tf.logging.INFO)

BUCKET = None  # set from task.py
PATTERN = 'of' # gets all files

# Determine CSV, label, and key columns
CSV_COLUMNS = 'hashname,chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,sugarpercent,pricepercent,winpercent'.split(',')
LABEL_COLUMN = 'winpercent'
KEY_COLUMN = 'hashname'

# Set default values for each CSV column
DEFAULTS = [['nokey'],[0], [0], [0], [0], [0], [0], [0], [0], [0],  [0.0], [0.0], [0.0]]

# Define some hyperparameters
TRAIN_STEPS = 10000
EVAL_STEPS = None
BATCH_SIZE = 512
NEMBEDS = 3
NNSIZE = [64, 16, 4]

# Create an input function reading a file using the Dataset API
# Then provide the results to the Estimator API
def read_dataset(prefix, mode, batch_size):
    def _input_fn():
        def decode_csv(value_column):
            columns = tf.decode_csv(value_column, record_defaults=DEFAULTS)
            features = dict(zip(CSV_COLUMNS, columns))
            label = features.pop(LABEL_COLUMN)
            return features, label
        
        # Use prefix to create file path
        file_path = 'gs://{}/candy/preproc/{}*{}*'.format(BUCKET, prefix, PATTERN)

        # Create list of files that match pattern
        file_list = tf.gfile.Glob(file_path)

        # Create dataset from file list
        dataset = (tf.data.TextLineDataset(file_list)  # Read text file
                    .map(decode_csv))  # Transform each elem by applying decode_csv fn
      
        if mode == tf.estimator.ModeKeys.TRAIN:
            num_epochs = None # indefinitely
            dataset = dataset.shuffle(buffer_size = 10 * batch_size)
        else:
            num_epochs = 1 # end-of-input after this
 
        dataset = dataset.repeat(num_epochs).batch(batch_size)
        return dataset.make_one_shot_iterator().get_next() #?
    return _input_fn

# Define feature columns
def get_wide_deep():
    # Define column types
  chocolate,fruity,caramel,peanutyalmondy,nougat,crispedricewafer,hard,bar,pluribus,sugarpercent,pricepercent = \
         [\
          tf.feature_column.categorical_column_with_vocabulary_list('chocolate', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('fruity', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('caramel', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('peanutyalmondy', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('nougat', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('crispedricewafer', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('hard', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('bar', [0, 1]),
          tf.feature_column.categorical_column_with_vocabulary_list('pluribus', [0, 1]),
          tf.feature_column.numeric_column('sugarpercent'),
          tf.feature_column.numeric_column('pricepercent')
         ]

  # Discretize
  sugar_buckets = tf.feature_column.bucketized_column(sugarpercent, 
                      boundaries=np.arange(0.2,0.8,.2).tolist())
  price_buckets = tf.feature_column.bucketized_column(pricepercent, 
                      boundaries=np.arange(0.2,0.8,.2).tolist())

  # Sparse columns are wide, have a linear relationship with the output
  wide = [\
          chocolate,
          fruity,
          #caramel,
          peanutyalmondy,
          #nougat,
          crispedricewafer,
          hard,
          bar,
          #pluribus,
          #sugar_buckets,
          price_buckets
         ]

  # Feature cross all the wide columns and embed into a lower dimension
  crossed = tf.feature_column.crossed_column(wide, hash_bucket_size=20000)
  embed = tf.feature_column.embedding_column(crossed, 3)

  # Continuous columns are deep, have a complex relationship with the output
  deep = [\
          #sugarpercent,
          #pricepercent,
          embed]
  return wide, deep

# Create serving input function to be able to serve predictions later using provided inputs
def serving_input_fn():
    feature_placeholders = {
        KEY_COLUMN: tf.placeholder_with_default(tf.constant(['nokey']), [None]),
        'chocolate': tf.placeholder(tf.int64, [None]),
        'fruity': tf.placeholder(tf.int64, [None]),
        #'caramel': tf.placeholder(tf.int64, [None]),
        'peanutyalmondy': tf.placeholder(tf.int64, [None]),
        #'nougat': tf.placeholder(tf.int64, [None]),
        'crispedricewafer': tf.placeholder(tf.int64, [None]),
        'hard': tf.placeholder(tf.int64, [None]),
        'bar': tf.placeholder(tf.int64, [None]),
        #'pluribus': tf.placeholder(tf.int64, [None]),
        #'sugarpercent': tf.placeholder(tf.float32, [None]),
        'pricepercent': tf.placeholder(tf.float32, [None])
    }
    features = {
        key: tf.expand_dims(tensor, -1)
        for key, tensor in feature_placeholders.items()
    }
    return tf.estimator.export.ServingInputReceiver(features, feature_placeholders)

# create metric for hyperparameter tuning
def my_rmse(labels, predictions):
    pred_values = predictions['predictions']
    return {'rmse': tf.metrics.root_mean_squared_error(labels, pred_values)}

# Create estimator to train and evaluate
def train_and_evaluate(output_dir):
    tf.summary.FileWriterCache.clear() # ensure filewriter cache is clear for TensorBoard events file
    wide, deep = get_wide_deep()
    EVAL_INTERVAL = 300 # seconds

    run_config = tf.estimator.RunConfig(save_checkpoints_secs = EVAL_INTERVAL,
                                        keep_checkpoint_max = 3)
    
    estimator = tf.estimator.DNNLinearCombinedRegressor(
        model_dir = output_dir,
        linear_feature_columns = wide,
        dnn_feature_columns = deep,
        dnn_hidden_units = NNSIZE,
        config = run_config)
    
    # illustrates how to add an extra metric
    estimator = tf.contrib.estimator.add_metrics(estimator, my_rmse)
    # for batch prediction, you need a key associated with each instance
    estimator = tf.contrib.estimator.forward_features(estimator, KEY_COLUMN)
    
    train_spec = tf.estimator.TrainSpec(
        input_fn = read_dataset('train', tf.estimator.ModeKeys.TRAIN, BATCH_SIZE),
        max_steps = TRAIN_STEPS)
    
    exporter = tf.estimator.LatestExporter('exporter', serving_input_fn, exports_to_keep=None)

    eval_spec = tf.estimator.EvalSpec(
        input_fn = read_dataset('eval', tf.estimator.ModeKeys.EVAL, 2**15),  # no need to batch in eval
        steps = EVAL_STEPS,
        start_delay_secs = 60, # start evaluating after N seconds
        throttle_secs = EVAL_INTERVAL,  # evaluate every N seconds
        exporters = exporter)
    tf.estimator.train_and_evaluate(estimator, train_spec, eval_spec)
