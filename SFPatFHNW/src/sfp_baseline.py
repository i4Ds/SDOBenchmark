from model_utils.model_runner import ModelRunner
from keras.layers import Lambda, Flatten, Dense

mr = ModelRunner('sfp_baseline_v1') # cuda_visible_devices="", log_device_placement=False

# optimal fixed point prediction, is median of full train data (which minimizes mae)
# Always predict the same small flare ("B5")
PREDICT_VALUE = 5.29411764705883E-07

main_input = mr.init_input(#sample_data_only = True,
              augment_data  = False)

layers = Lambda(lambda x: x * 0, name='lambda_to0')(main_input)
layers = Flatten()(layers)
layers = Dense(1, activation='linear')(layers)
layers = Lambda(lambda x: x + PREDICT_VALUE, name='lambda_toPREDICT_VALUE', output_shape=lambda s: (s[0], 1))(layers)

mr.train(layers, epochs=0, batch_size=1, dont_train=True, save_wheights=False, eval_to_console_after_init=False, eval_to_console_after_train=True)
