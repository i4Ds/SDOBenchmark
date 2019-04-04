DATA_FOLDER = "../../data/";

disp("FULL DATA SET");
full_train_flux = csvread(DATA_FOLDER + "full/train/meta_data.csv",1,3);
full_test_flux = csvread(DATA_FOLDER + "full/test/meta_data.csv",1,3);

prediction = median(full_train_flux);
mae = sum(abs(full_test_flux - prediction)) / length(full_test_flux);

disp("train size: " + length(full_train_flux));
disp("train max:  " + max(full_train_flux));
disp("train min:  " + min(full_train_flux));
disp(" ");
disp("test size:  " + length(full_test_flux));
disp("test max:   " + max(full_test_flux));
disp("test min:   " + min(full_test_flux));
disp(" ");
disp("median:     " + prediction);
disp("mae:        " + mae);

disp(" ");
disp("SAMPLE DATA SET");
sample_train_flux = csvread(DATA_FOLDER + "sample/train/meta_data.csv",1,3);
sample_test_flux = csvread(DATA_FOLDER + "sample/test/meta_data.csv",1,3);

prediction = median(sample_train_flux);
mae = sum(abs(sample_test_flux - prediction)) / length(sample_test_flux);

disp("train size: " + length(sample_train_flux));
disp("train max:  " + max(sample_train_flux));
disp("train min:  " + min(sample_train_flux));
disp(" ");
disp("test size:  " + length(sample_test_flux));
disp("test max:   " + max(sample_test_flux));
disp("test min:   " + min(sample_test_flux));
disp(" ");
disp("median:     " + prediction);
disp("mae:        " + mae);
