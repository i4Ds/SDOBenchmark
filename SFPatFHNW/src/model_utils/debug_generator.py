import os

from keras_generator import SDOBenchmarkGenerator

model_name = 'simple_cnn_model'
base_path = '../../data/sample'
params = {'dim': (4, 256, 256, 2),
          'batch_size': 32,
          'channels': ['magnetogram', '1700'],
          'data_format': 'channels_first',
          'shuffle': True,
          'augment': False,
          'include_date': False,
          'only_last_slice': True,
          'no_image_data': True}

training_generator = SDOBenchmarkGenerator(os.path.join(base_path, 'train'), **params)

sum = 0
for batch in training_generator:
    X, y = batch
    print(f"X shape: {X[0].shape}")
    print(f"y shape: {y.shape}")
    print(batch)
    sum += 1

print(f"batches: {sum}")
print(
    f"{training_generator.imagesLoaded} / {training_generator.imagesExpected} ({(training_generator.imagesLoaded * 100) / training_generator.imagesExpected:.2f}%)")

# (X, y) = training_generator[20]
# print("\nDATA:")
# print(f'len(X) = {len(X)}')
# print(X)
# print("\nLABELS:")
# print(y)
# print("\n")
