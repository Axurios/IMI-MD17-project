import os

def list_directories_with_code(root_dir, extensions=['.py', '.java', '.cpp', '.c']):
    code_directories = []
    for dirpath, dirnames, filenames in os.walk(root_dir):
        for filename in filenames:
            if any(filename.endswith(ext) for ext in extensions):
                code_directories.append(dirpath)
                break
    return code_directories

def execute_sbatch_jsub(directory_list):

    basedir=os.getcwd()
    for directory in directory_list:
        # Remplacez cette ligne par la commande sbatch jsub avec le bon chemin du répertoire
        #print(f"Lancement de sbatch jsub dans {directory}")
        os.chdir(directory)
        os.system("sbatch jsub")
        os.chdir(basedir)

ddir=os.getcwd()

if __name__ == "__main__":
    #root_directory = "/chemin/du/repertoire/racine"  # Remplacez par votre répertoire racine
    root_directory = ddir
    directories_with_code = list_directories_with_code(root_directory)
    execute_sbatch_jsub(directories_with_code)
