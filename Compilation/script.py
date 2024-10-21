import shutil
import os


def modify_train_script(
    folder,
    features,
    max_degree,
    num_iterations,
    num_basis_functions,
    cutoff,
    num_train,
    num_valid,
    num_epochs,
    learning_rate,
    forces_weight,
    batch_size,
):
    with open(str(folder) + "/train_run.py", "r") as file:
        train_script = file.readlines()

    modified_script = []
    for line in train_script:
        if line.startswith("features"):
            modified_script.append(f"features = {features}\n")
        elif line.startswith("max_degree"):
            modified_script.append(f"max_degree = {max_degree}\n")
        elif line.startswith("num_iterations"):
            modified_script.append(f"num_iterations = {num_iterations}\n")
        elif line.startswith("num_basis_functions"):
            modified_script.append(f"num_basis_functions = {num_basis_functions}\n")
        elif line.startswith("cutoff"):
            modified_script.append(f"cutoff = {cutoff}\n")
        elif line.startswith("num_train"):
            modified_script.append(f"num_train = {num_train}\n")
        elif line.startswith("num_valid"):
            modified_script.append(f"num_valid = {num_valid}\n")
        elif line.startswith("num_epochs"):
            modified_script.append(f"num_epochs = {num_epochs}\n")
        elif line.startswith("learning_rate"):
            modified_script.append(f"learning_rate = {learning_rate}\n")
        elif line.startswith("forces_weight"):
            modified_script.append(f"forces_weight = {forces_weight}\n")
        elif line.startswith("batch_size"):
            modified_script.append(f"batch_size = {batch_size}\n")
        else:
            modified_script.append(line)

    with open(str(folder) + "/train_run.py", "w") as file:
        file.writelines(modified_script)


L_num_trains = [100 * k for k in range(1, 10, 2)]
L_num_basis_functions = [32]
L_cutoff = [3, 4]
L_batch_size = [10, 30, 50]
L_num_valid = [k // 5 for k in L_num_trains]
L_learning_rate = [0.01]
L_forces_weight = [0.01, 0.1]
L_features = [32, 64]
max_degree = 2

source_file_train_run = "train_run.py"
data = "md17_ethanol.npz"

for features in L_features:
    for num_train in L_num_trains:
        for num_basis_functions in L_num_basis_functions:
            for cutoff in L_cutoff:
                for batch_size in L_batch_size:
                    for learning_rate in L_learning_rate:
                        for forces_weight in L_forces_weight:

                            folder_path = (
                                "./"
                                + str(features)
                                + "_"
                                + str(num_train)
                                + "_"
                                + str(num_basis_functions)
                                + "_"
                                + str(cutoff)
                                + "_"
                                + str(batch_size)
                                + "_"
                                + str(num_train // 5)
                                + "_"
                                + str(int(learning_rate * 1000))
                                + "_"
                                + str(int(forces_weight * 100))
                            )

                            if not os.path.exists(folder_path):
                                os.makedirs(folder_path)

                            shutil.copy(source_file_train_run, folder_path)
                            # shutil.copy(data, folder_path)

                            modify_train_script(
                                folder=folder_path,
                                features=features,
                                max_degree=max_degree,
                                num_iterations=3,
                                num_basis_functions=num_basis_functions,
                                cutoff=3.0,
                                num_train=num_train,
                                num_valid=num_train // 5,
                                num_epochs=300,
                                learning_rate=learning_rate,
                                forces_weight=forces_weight,
                                batch_size=batch_size,
                            )
