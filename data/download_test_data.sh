mkdir -p $1"/test_data/"
mkdir -p $1"/meshes/"
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1nPV-L2oG0g4h1lJ5hszsGpRCul1BcwTU' -O $1"ycb_meshes.zip"
unzip -n -q $1"ycb_meshes.zip" -d $1"/meshes/"
rm $1"ycb_meshes.zip"
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1eQnMYiOJk1I26L6arWKYvINiInh3EyIe' -O $1"egad_validation_meshes.zip"
unzip -n -q $1"egad_validation_meshes.zip" -d $1"/meshes/"
rm $1"egad_validation_meshes.zip"
wget --no-check-certificate -r 'https://drive.google.com/uc?export=download&id=1bqhKCOdtxqrsRjJ6bvI5E3MF6NUx00Li' -O $1"graspit_test_grasps_simulated_annealing.zip"
unzip -n -q $1"graspit_test_grasps_simulated_annealing.zip" -d $1"/test_data/"
rm $1"graspit_test_grasps_simulated_annealing.zip"